For activating the visualizer, with any example, just pass the option
--SimulatorImplementationType=ns3::VisualSimulatorImpl to it, assuming the
script uses ns-3's command line parser (class CommandLine), and add 'visualizer'
as a module dependency to that program.

Alternatively, run the example with ns3 run adding the --visualize option.
For example:

./ns3 run wifi-simple-adhoc-grid --visualize
