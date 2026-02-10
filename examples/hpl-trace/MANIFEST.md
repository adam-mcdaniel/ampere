# Experiment directory overview

The result directory of this measurement should contain the following files:

   1. Files that should be present even if the measurement aborted:

      * `MANIFEST.md`           This manifest file.
      * `scorep.cfg`            Listing of used environment variables.

   2. Files that will be created by subsystems of the measurement core:

      * Tracing:

        * `traces.otf2`         OTF2 anchor file.
        * `traces.def`          OTF2 global definitions file.
        * `traces/`             Sub-directory containing per location trace
                                data.

# List of Score-P variables that were explicitly set for this measurement

The complete list of Score-P variables used, incl. current default values,
can be found in `scorep.cfg`.

    SCOREP_ENABLE_PROFILING
    SCOREP_ENABLE_TRACING
    SCOREP_ENABLE_UNWINDING
    SCOREP_VERBOSE
    SCOREP_TOTAL_MEMORY
    SCOREP_PAGE_SIZE
    SCOREP_EXPERIMENT_DIRECTORY
    SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY
    SCOREP_MACHINE_NAME
    SCOREP_EXECUTABLE
    SCOREP_FORCE_CFG_FILES
    SCOREP_TIMER
    SCOREP_PROFILING_TASK_EXCHANGE_NUM
    SCOREP_PROFILING_MAX_CALLPATH_DEPTH
    SCOREP_PROFILING_BASE_NAME
    SCOREP_PROFILING_FORMAT
    SCOREP_PROFILING_ENABLE_CLUSTERING
    SCOREP_PROFILING_CLUSTER_COUNT
    SCOREP_PROFILING_CLUSTERING_MODE
    SCOREP_PROFILING_CLUSTERED_REGION
    SCOREP_PROFILING_ENABLE_CORE_FILES
    SCOREP_TRACING_USE_SION
    SCOREP_TRACING_MAX_PROCS_PER_SION_FILE
    SCOREP_TRACING_CONVERT_CALLING_CONTEXT_EVENTS
    SCOREP_FILTERING_FILE
    SCOREP_SUBSTRATE_PLUGINS
    SCOREP_SUBSTRATE_PLUGINS_SEP
    SCOREP_LIBWRAP_PATH
    SCOREP_LIBWRAP_ENABLE
    SCOREP_LIBWRAP_ENABLE_SEP
    SCOREP_METRIC_RUSAGE
    SCOREP_METRIC_RUSAGE_PER_PROCESS
    SCOREP_METRIC_RUSAGE_SEP
    SCOREP_METRIC_PLUGINS
    SCOREP_METRIC_PLUGINS_SEP
    SCOREP_METRIC_PERF
    SCOREP_METRIC_PERF_PER_PROCESS
    SCOREP_METRIC_PERF_SEP
    SCOREP_METRIC_AROCM_SMI_PLUGIN
    SCOREP_METRIC_CORETEMP_PLUGIN
    SCOREP_SAMPLING_EVENTS
    SCOREP_SAMPLING_SEP
    SCOREP_TOPOLOGY_PLATFORM
    SCOREP_TOPOLOGY_PROCESS
    SCOREP_HIP_ENABLE
    SCOREP_HIP_ACTIVITY_BUFFER_SIZE
    SCOREP_MEMORY_RECORDING
    SCOREP_IO_POSIX
