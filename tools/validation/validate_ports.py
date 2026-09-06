"""Run the workspace's software port gates and retain reproducible evidence.

Run in conda MIR. No ADB command, device claim, flash or physical actuation is
performed. Full trained-stream failures remain failures even when the smaller
component fixtures pass. --phase selects a partial run, never a full pass.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PHASES = ('native', 'references', 'trained', 'precise', 'android', 'firmware')
REPOSITORIES = ('mir-core', 'mir-desktop-app', 'mir-embedded-ai', 'mir-embedded-pp',
                'mir-android-app', 'mir-train-hpc', 'mir-embedded-hmi')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): value.update(block)
    return value.hexdigest()


def source_snapshot():
    """Include tracked and untracked source; omit build output and credentials."""
    suffixes = {'.py', '.rs', '.c', '.cc', '.cpp', '.h', '.hpp', '.kt', '.kts',
                '.toml', '.lock', '.yml', '.yaml', '.json', '.cmake', '.gradle', '.ini'}
    result = {}
    for name in REPOSITORIES:
        repo = ROOT/name
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        files = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=repo).decode().split('\0')
        hashes = {}
        for filename in sorted(set(files)):
            path = repo/filename
            if path.is_file() and (path.suffix in suffixes or path.name == 'CMakeLists.txt'):
                hashes[filename] = sha256(path)
        result[name] = dict(head=head, files=hashes,
            source_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
    return result


def command_plan(output, scratch, legacy_run=None, *, precise_android=False):
    py = sys.executable
    core = ROOT/'mir-core'
    pp = ROOT/'mir-embedded-pp'
    desktop = ROOT/'mir-desktop-app'
    android = ROOT/'mir-android-app'
    dsp = core/'native-dsp'
    litert = ROOT/'mir-embedded-ai/native-litert'
    capi = desktop/'native-runtime-capi'
    pp_build = pp/'build/ports-remediation'
    shared_pp = pp_build/'libmir_embedded_pp_host.so'
    replay = capi/'build/mir_model_replay'
    router_fixture = scratch/'router-reference.json'
    onnx = scratch/'onnx/nativePorts'
    yamnet = scratch/'yamnet/nativePorts'
    plan = {phase: [] for phase in PHASES}

    def add(phase, name, argv, cwd=ROOT, extra=None):
        plan[phase].append(dict(name=name, argv=list(map(str, argv)), cwd=str(cwd), environment=extra or {}))

    for name, source, build in [('dsp', dsp, dsp/'build'), ('litert', litert, litert/'build'), ('postprocess', pp, pp_build)]:
        add('native', name+'-configure', ['cmake', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release'])
        add('native', name+'-build', ['cmake', '--build', build, '--parallel', '2'])
    add('native', 'postprocess-ctest', ['ctest', '--test-dir', pp_build, '--output-on-failure'])
    for name, crate in [('dsp', dsp/'rust'), ('litert', litert/'rust'), ('postprocess', pp/'rust'),
                        ('runtime', desktop/'native-runtime'), ('capi', capi)]:
        add('native', name+'-clippy', ['cargo', 'clippy', '--locked', '--all-targets', '--', '-D', 'warnings'], crate)
        add('native', name+'-tests', ['cargo', 'test', '--locked', '--all-targets'], crate)
    add('native', 'dsp-replay-build', ['cargo', 'build', '--locked', '--release', '--examples'], dsp/'rust')
    add('native', 'runtime-python-build', [py, '-m', 'maturin', 'develop', '--release', '--locked'], desktop/'native-runtime')
    add('native', 'capi-release', ['cargo', 'build', '--release', '--locked'], capi)
    add('native', 'capi-configure', ['cmake', '-S', capi, '-B', capi/'build',
        f'-DMIR_RUNTIME_LIBRARY={capi}/target/release/libmir_native_runtime_capi.so'])
    add('native', 'capi-build', ['cmake', '--build', capi/'build', '--parallel', '2'])
    add('native', 'engine-tests', ['cargo', 'test', '--locked', '--all-targets'], desktop/'native-engine')
    add('native', 'engine-actual-decoder', ['cargo', 'test', '--locked', '--all-targets', '--', '--ignored', '--skip', 'every_hot_route_matches_an_independent_uninterrupted_stage'], desktop/'native-engine')
    add('native', 'engine-build', ['cargo', 'build', '--locked'], desktop/'native-engine')
    add('native', 'workflows-build', ['cargo', 'build', '--locked'], desktop/'native-workflows')
    add('native', 'workflows-clippy', ['cargo', 'clippy', '--locked', '--all-targets', '--', '-D', 'warnings'], desktop/'native-workflows')
    add('native', 'workflows-tests', ['cargo', 'test', '--locked', '--all-targets'], desktop/'native-workflows')
    add('native', 'causal-classifier-replay-build', ['cargo', 'build', '--locked', '--example', 'causal_classifier_replay'], desktop/'native-runtime')
    add('native', 'desktop-tests', ['cargo', 'test', '--locked', '--all-targets'], desktop/'desktop-rust')
    qt = desktop/'desktop-cpp'
    add('native', 'qt-configure', ['cmake', '-S', qt, '-B', qt/'build/ports-remediation', '-DCMAKE_BUILD_TYPE=Release'])
    add('native', 'qt-build', ['cmake', '--build', qt/'build/ports-remediation', '--parallel', '2'])
    add('native', 'qt-tests', ['ctest', '--test-dir', qt/'build/ports-remediation', '--output-on-failure'], extra={'QT_QPA_PLATFORM':'offscreen'})

    add('references', 'resampling', [py, dsp/'tools/resampling_parity.py', '--cpp-replay', dsp/'build/mir_resampler_replay',
        '--rust-replay', dsp/'rust/target/release/examples/resampler_replay', '--library', dsp/'build/libmir_dsp.so',
        '--output', output/'resampling.json'])
    for example, fixture in [('check_replay', 'port_reference.json'), ('check_dance', 'dance_port_reference.json')]:
        add('references', example, ['cargo', 'run', '--locked', '--example', example, '--', shared_pp, pp/'test/fixtures'/fixture], pp/'rust')
    add('references', 'offline-options-replay', ['cargo', 'run', '--locked', '--example', 'check_replay', '--', shared_pp, pp/'test/fixtures/offline_options_reference.json'], pp/'rust')
    add('references', 'offline-options-reference-export', [py, pp/'tools/export_offline_options_fixtures.py', scratch/'offline_options_reference.json'])
    add('references', 'offline-options-reference-compare', ['cmp', pp/'test/fixtures/offline_options_reference.json', scratch/'offline_options_reference.json'])
    add('references', 'postprocess-reference-freshness-export', [py, pp/'tools/export_port_fixtures.py', scratch/'port_reference.json'])
    add('references', 'postprocess-reference-freshness-compare', ['cmp', pp/'test/fixtures/port_reference.json', scratch/'port_reference.json'])
    add('references', 'particle-filter-dual-rng', [py, pp/'tools/check_particle_filter_contract.py',
        '--replay', pp_build/'particle_filter_reference_contract',
        '--production', pp_build/'particle_filter_production_contract',
        '--output', output/'particle-filter-dual-rng.json'])
    add('references', 'particle-filter-contract-regressions', [py, '-m', 'pytest', '-q',
        core/'tests/test_postprocessing_particle_filter.py', core/'tests/test_particle_filter_rng_contract.py',
        pp/'test/python/test_particle_filter_contract.py', '-k', 'not test_frozen_dual_rng_manifest'],
        extra={'MIR_PF_CONTRACT_BUILD': str(pp_build)})
    add('references', 'router-reference', [py, core/'tools/validation/export_router_fixtures.py', android/'app/src/main/assets/models/catalog.json', router_fixture])
    add('references', 'router-cpp', [py, capi/'tests/check_router.py', capi/'build/mir_router_replay', router_fixture, output/'router.json'])
    add('references', 'logspect-cpp', [py, capi/'tests/check_frontend.py', replay, desktop/'native-runtime/tests/fixtures', output/'logspect.json'])
    add('references', 'spectnt-model', [py, '-m', 'pytest', '-q', core/'tests/test_spectnt.py'])
    suites = [core/'tests/test_native_beatnet.py', core/'tests/test_precise_streaming.py', core/'tests/test_native_batch.py',
              core/'tests/test_native_classifier.py', core/'tests/test_native_legacy_bock.py', core/'tests/test_native_batch_frontend.py',
              core/'tests/test_native_classifier_frontend.py',
              core/'tools/validation/test_long_models.py', ROOT/'mir-train-hpc/tests/test_native_classifier_frontend.py']
    add('references', 'onnx-source-and-cpp-replay', [py, '-m', 'pytest', '-q', '--import-mode=importlib', '-p', 'port_validation_plugin', *suites],
        extra={'MIR_PORT_FIXTURES': str(onnx), 'MIR_CPP_REPLAY': str(replay)})
    add('references', 'yamnet-source-and-cpp-replay', [py, '-m', 'pytest', '-q', '--import-mode=importlib', '-p', 'port_validation_plugin',
        ROOT/'mir-train-hpc/tests/test_native_yamnet_frontend.py'],
        extra={'MIR_PORT_FIXTURES': str(yamnet), 'MIR_CPP_REPLAY': str(replay)})
    add('references', 'promoted-classifiers', [py, core/'tools/validation/check_promoted_classifiers.py',
        '--output', output/'classifiers', '--cpp-replay', replay])
    add('references', 'causal-classifier-source', [py, core/'tools/validation/check_native_causal_classifier.py', scratch/'causal-classifier'])
    add('references', 'promoted-causal-routing', [py, core/'tools/validation/check_native_promoted_routing.py', scratch/'promoted-routing'])
    add('references', 'desktop-routing', [py, core/'tools/validation/check_native_desktop_routing.py',
        scratch/'causal-classifier/logspect-running_peak-22050.input.json', scratch/'desktop-routing'])
    add('references', 'desktop-dance-dispatch-export', [py, core/'tools/validation/export_desktop_dance_dispatch.py', scratch/'dance_dispatch.json'])
    add('references', 'desktop-dance-dispatch-freshness', ['cmp', scratch/'dance_dispatch.json', desktop/'native-engine/fixtures/dance_dispatch.json'])
    add('references', 'native-experiment-workflows', [py, core/'tools/validation/check_native_workflows.py', '--output', scratch/'experiment-workflows'])
    add('references', 'native-device-workflows', [py, core/'tools/validation/check_native_devices.py', '--output', scratch/'device-workflows'])
    add('references', 'native-playback-math', [py, core/'tools/validation/check_native_playback_math.py', '--output', scratch/'playback-math'])
    add('references', 'native-playback-integration', [py, core/'tools/validation/check_native_playback.py', '--output', scratch/'playback-integration'])
    add('references', 'native-inferred-trials', [py, core/'tools/validation/check_native_trial_runtime.py', '--runtime-session', scratch/'desktop-routing/session.json', '--output', scratch/'inferred-trials'])
    add('references', 'native-runtime-binding', [py, core/'tools/validation/check_native_runtime_binding.py', '--classifier-replay', scratch/'promoted-routing/yamnet-fold-0/input.json', '--native-session', scratch/'desktop-routing/session.json', '--output', scratch/'runtime-binding'])
    add('references', 'input-label-freshness', [py, core/'tools/validation/export_native_input_names.py', '--output', desktop/'native-workflows/defaults/input_names.json', '--check'])

    add('android', 'android-jvm-and-apk', [android/'gradlew', '--no-daemon', ':app:testDebugUnitTest',
        ':haptic-common:testDebugUnitTest', ':wear:testDebugUnitTest', ':app:assembleDebug', ':app:assembleDebugAndroidTest',
        ':wear:assembleDebug', f'-PnativePortFixtures={scratch}/onnx',
        *([f'-PprecisePortFixtures={scratch}/precise/phone'] if precise_android else [])], android)
    add('trained', 'promoted-long-streams', [py, core/'tools/validation/check_promoted_streams.py', '--output', scratch/'trained',
        '--cpp-replay', replay, '--postprocessor-library', shared_pp, '--frames', '4096', '--state-atol', '0.00005'])
    add('precise', 'portable-deployment-streams', [py, core/'tools/validation/check_precise_streams.py',
        '--output', scratch/'precise', '--cpp-replay', replay, '--postprocessor-library', shared_pp,
        '--legacy-run', legacy_run or scratch/'trained', '--frames', '4096'])
    add('firmware', 'pp-firmware', ['pio', 'run', '-e', 'seeed_xiao_esp32s3', '-e', 'seeed_xiao_esp32s3_plus', '-e', 'seeed_xiao_esp32s3_dance_one'], pp)
    hmi = ROOT/'mir-embedded-hmi'
    add('firmware', 'hmi-configure', ['cmake', '-S', hmi, '-B', hmi/'build/ports-remediation', '-DCMAKE_BUILD_TYPE=Release'])
    add('firmware', 'hmi-build', ['cmake', '--build', hmi/'build/ports-remediation', '--parallel', '2'])
    add('firmware', 'hmi-tests', ['ctest', '--test-dir', hmi/'build/ports-remediation', '--output-on-failure'])
    add('firmware', 'hmi-firmware', ['pio', 'run', '-e', 'seeed_xiao_esp32s3'], hmi)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory (must not already contain a report)')
    parser.add_argument('--scratch', type=Path, required=True, help='Graph/reference cache outside tracked sources; Android-only runs reuse a completed reference cache')
    parser.add_argument('--phase', choices=PHASES, action='append')
    parser.add_argument('--legacy-run', type=Path,
        help='Completed original float32 trained-stream evidence for a precise-only run; the original compatibility gate remains separate')
    parser.add_argument('--java-home', type=Path, help='Java 17 JDK; otherwise locate an installed Java 17 automatically')
    parser.add_argument('--list', action='store_true', help='Print commands without executing them')
    args = parser.parse_args()
    output, scratch = args.output.resolve(), args.scratch.resolve()
    selected = set(args.phase or PHASES)
    plan = command_plan(output, scratch, args.legacy_run,
        precise_android='precise' in selected or (scratch/'precise/phone/precisePorts/index.json').is_file())
    if args.list:
        for phase in PHASES:
            if phase in selected:
                for entry in plan[phase]: print(phase, entry['cwd'], shlex.join(entry['argv']))
        return 0
    if Path(sys.prefix).name != 'MIR': parser.error('Run this command in the conda environment MIR')
    if (output/'report.json').exists(): parser.error('Use a new output directory; existing evidence is never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    (output/'logs').mkdir(exist_ok=True)
    for directory in ('onnx/nativePorts', 'yamnet/nativePorts'):
        if 'references' in selected and (scratch/directory/'index.json').exists():
            parser.error('Use a fresh scratch directory when regenerating reference fixtures')
    if 'android' in selected and 'references' not in selected and not (scratch/'onnx/nativePorts/index.json').is_file():
        parser.error('An Android-only run requires --scratch pointing to a completed reference run')
    environment = dict(os.environ)
    environment.update(PYTHONPATH=os.pathsep.join(str(ROOT/name) for name in
        ('mir-core/tools', 'mir-core', 'mir-train-hpc', 'mir-desktop-app/src')),
        MIR_RUN_MODEL_AUTHENTICITY_TESTS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2',
        MIR_EMBEDDED_PP_HOST_LIBRARY=str(ROOT/'mir-embedded-pp/build/ports-remediation/libmir_embedded_pp_host.so'),
        MIR_LITERT_LIBRARY=str(ROOT/'mir-embedded-ai/native-litert/build/libmir_litert.so'))
    # Read JDK metadata rather than accepting a machine's Java 11 default.
    candidates = ([args.java_home] if args.java_home else
                  ([Path(environment['JAVA_HOME'])] if environment.get('JAVA_HOME') else []) +
                  sorted(Path('/usr/lib/jvm').glob('*17*')))
    java = next((p for p in candidates if (p/'release').is_file() and
                 any(line.startswith('JAVA_VERSION="17.') for line in (p/'release').read_text().splitlines())), None)
    if 'android' in selected and java is None:
        parser.error('Android requires an installed Java 17 JDK; provide --java-home')
    if java is not None: environment['JAVA_HOME'] = str(java.resolve())
    result = dict(schema='mir.workspace-port-validation/v1', started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        selected_phases=[p for p in PHASES if p in selected], required_phases=list(PHASES),
        all_required_phases_selected=selected == set(PHASES),
        hardware_status='not_run_requires_separate_coordination_and_device_evidence',
        musical_accuracy_status='not_certified_by_numerical_parity', commands=[], phases={})
    result['environment'] = dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), machine=platform.machine(), java_home=environment.get('JAVA_HOME'),
        packages={name:metadata.version(name) for name in ('numpy', 'torch', 'onnx', 'onnxruntime', 'soxr')})
    result['sources_before'] = source_snapshot()
    report = output/'report.json'
    try:
        for phase in PHASES:
            if phase not in selected:
                result['phases'][phase] = 'not_run'
                continue
            result['phases'][phase] = 'running'
            report.write_text(json.dumps(result, indent=2)+'\n')
            phase_failed = False
            for entry in plan[phase]:
                record = {**entry, 'phase': phase, 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
                log = output/'logs'/f'{phase}-{entry["name"]}.log'
                print(f'[{phase}] {entry["name"]}', flush=True)
                started = time.monotonic()
                try:
                    with log.open('w') as stream:
                        completed = subprocess.run(entry['argv'], cwd=entry['cwd'],
                            env={**environment, **entry['environment']}, stdout=stream, stderr=subprocess.STDOUT)
                    record['exit_code'] = completed.returncode
                except OSError as error:
                    log.write_text(str(error)+'\n')
                    record['exit_code'] = 127
                record.update(seconds=time.monotonic()-started, log=str(log), log_sha256=sha256(log))
                result['commands'].append(record)
                report.write_text(json.dumps(result, indent=2)+'\n')
                if record['exit_code']:
                    phase_failed = True
                    print(f'FAILED {entry["name"]}: {log}', flush=True)
                    if phase == 'native': break  # remaining commands require successful builds
            result['phases'][phase] = 'fail' if phase_failed else 'pass'
            report.write_text(json.dumps(result, indent=2)+'\n')
            if phase == 'native' and phase_failed:
                for remaining in PHASES[1:]:
                    result['phases'][remaining] = 'not_run_dependency_failed' if remaining in selected else 'not_run'
                break
    finally:
        result['evidence_files'] = {}
        for name, source in [('onnx-replay.json', scratch/'onnx/nativePorts/index.json'),
                             ('yamnet-replay.json', scratch/'yamnet/nativePorts/index.json'),
                             ('trained-streams.json', scratch/'trained/report.json'),
                             ('precise-streams.json', scratch/'precise/report.json'),
                             ('precise-phone-index.json', scratch/'precise/phone/precisePorts/index.json')]:
            if source.is_file():
                shutil.copyfile(source, output/name)
                result['evidence_files'][name] = sha256(output/name)
        result['native_binaries'] = {str(p.relative_to(ROOT)):sha256(p) for p in (
            ROOT/'mir-core/native-dsp/build/libmir_dsp.so',
            ROOT/'mir-embedded-pp/build/ports-remediation/libmir_embedded_pp_host.so',
            ROOT/'mir-embedded-ai/native-litert/build/libmir_litert.so',
            ROOT/'mir-desktop-app/native-runtime-capi/target/release/libmir_native_runtime_capi.so',
            ROOT/'mir-desktop-app/native-runtime-capi/build/mir_model_replay') if p.is_file()}
        result['sources_after'] = source_snapshot()
        result['sources_unchanged_during_run'] = result['sources_before'] == result['sources_after']
        result['selected_checks_passed'] = all(result['phases'].get(p) == 'pass' for p in selected)
        result['full_software_gate_passed'] = selected == set(PHASES) and result['selected_checks_passed'] and result['sources_unchanged_during_run']
        result['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('phases', 'sources_unchanged_during_run', 'selected_checks_passed', 'full_software_gate_passed')}, indent=2))
    return 0 if result['selected_checks_passed'] and result['sources_unchanged_during_run'] else 1


if __name__ == '__main__': raise SystemExit(main())
