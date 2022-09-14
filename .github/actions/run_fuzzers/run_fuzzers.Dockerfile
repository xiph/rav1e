FROM gcr.io/oss-fuzz-base/clusterfuzzlite-run-fuzzers:v1
ENV FUZZER_ARGS="-rss_limit_mb=2560 -timeout=600"
RUN sed -i 's/timeout=25/timeout=600/' ${OSS_FUZZ_ROOT}/infra/cifuzz/base_runner_utils.py
RUN sed -i '/return self.config.report_timeouts/a \    if testcase.startswith("slow-unit-"): return True' ${OSS_FUZZ_ROOT}/infra/cifuzz/fuzz_target.py
RUN sed -i 's/timeout=100/timeout=2400/;s/TIMEOUT=1h/TIMEOUT=5h/' /usr/local/bin/coverage
RUN find /usr/local -name constants.py -exec sed -i 's/DEFAULT_TIMEOUT_LIMIT = 25/DEFAULT_TIMEOUT_LIMIT = 600/' {} +
RUN find /usr/local -name libfuzzer.py -exec sed -i 's/crash.oom.timeout.leak/&|slow-unit/' {} +
