sigrity::open document {!}
sigrity::open workflow -product {PowerSI} -workflowkey {extraction} {!}
sigrity::save {D:\A0607\PowerSI\CFP4\CFP4.spd} {!}
sigrity::add net {!}
sigrity::add net {!}
sigrity::update net name {VCC} {NewEntity} {!}
sigrity::update net name {GND} {NewEntity(1)} {!}
sigrity::move net {PowerNets} {VCC} {!}
sigrity::move net {GroundNets} {GND} {!}
sigrity::update net selected 1 {VCC} {!}
sigrity::update net selected 1 {PowerNets} {!}
sigrity::update net selected 1 {GND} {!}
sigrity::update net selected 1 {GroundNets} {!}
sigrity::update layer layer_name {Signal$Top} {Signal02} {!}
sigrity::delete layer {Signal01} {!}
sigrity::delete layer {Plane01} {!}
sigrity::delete layer {Medium01} {!}
sigrity::delete layer {Medium02} {!}
sigrity::update layer layer_name {Medium01} {Medium03} {!}
sigrity::update layer layer_name {Plane$GND} {Plane02} {!}
sigrity::update layer thickness 4.826e-05 {Signal$Top} {!}
sigrity::update layer thickness 3.048e-05 {Plane$GND} {!}
sigrity::update layer thickness 2.0320e-04 {Medium01} {!}
sigrity::update layer model_name {FR-4} {Medium01} {!}
sigrity::update layer conductivity 5.959000e+07 {Signal$Top} {!}
sigrity::update layer conductivity 5.959000e+07 {Plane$GND} {!}
sigrity::update layer dielectric_name {} {Signal$Top} {!}
sigrity::update layer dielectric_name {} {Plane$GND} {!}
sigrity::update layer Er {1} {Signal$Top} {!}
sigrity::update layer Er {4.5} {Plane$GND} {!}
sigrity::update layer loss_tangent {0} {Signal$Top} {!}
sigrity::update layer loss_tangent {0.035} {Plane$GND} {!}
sigrity::update layer trace_width 5.500e-05 {Signal$Top} {!}
sigrity::update dielectric_material {!}
sigrity::open LayoutView -layer {Plane$GND} {!}
sigrity::update Box {Box002}  -Net {GND} {!}
sigrity::update Box {Box002}  -LeftCornerCoorX {-0.120000} -LeftCornerCoorY {-0.120000} -Width {0.240000} -Height {0.240000} {!}
sigrity::open LayoutView -layer {Signal$Top} {!}
sigrity::add trace {-0.000745,-0.000255} {-0.000457,0.000033} {-0.000457,0.000433} {-0.000457,0.000733} {-0.000457,0.001213} {-0.000216,0.001454} {0.000026,0.001696} {0.000026,0.002123} {0.000026,0.002423} {0.000026,0.002723} {0.000005,0.002745} -Layer {Signal$Top} {!}
sigrity::add trace {0.000255,-0.000255} {0.000000,0.000000} {0.000000,0.000400} {0.000000,0.000700} {0.000000,0.001023} {0.000242,0.001265} {0.000483,0.001507} {0.000483,0.002123} {0.000483,0.002423} {0.000483,0.002723} {0.000505,0.002745} -Layer {Signal$Top} {!}
sigrity::open window port {!}
sigrity::add port {!}
sigrity::add port {!}
sigrity::add port {!}
sigrity::add port {!}
sigrity::close window port {!}
sigrity::check LayerSelection -ShowNode {1} {!}
sigrity::hook -port {Port1} -PositiveNode {Node01} {!}
sigrity::hook -port {Port2} -PositiveNode {Node011} {!}
sigrity::hook -port {Port3} -PositiveNode {Node012} {!}
sigrity::hook -port {Port4} -PositiveNode {Node022} {!}
sigrity::add net {!}
sigrity::add net {!}
sigrity::update net name {Sig1} {NewEntity(1)} {!}
sigrity::update net name {Sig2} {NewEntity} {!}
sigrity::update net color {255255000} {Sig1} {!}
sigrity::update net color {000000127} {Sig2} {!}
sigrity::update net selected 1 {Sig1} {!}
sigrity::update net selected 1 {Sig2} {!}
sigrity::update Trace {Trace010} -Net {Sig1} -Color {YELLOW} {!}
sigrity::update Trace {Trace09} -Net {Sig1} -Color {YELLOW} {!}
sigrity::update Trace {Trace08} -Net {Sig1} -Color {YELLOW} {!}
sigrity::update Trace {Trace07} -Net {Sig1} -Color {YELLOW} {!}
sigrity::update Trace {Trace06} -Net {Sig1} -Color {YELLOW} {!}
sigrity::update Trace {Trace05} -Net {Sig1} -Color {YELLOW} {!}
sigrity::update Trace {Trace04} -Net {Sig1} -Color {YELLOW} {!}
sigrity::update Trace {Trace02} -Net {Sig1} -Color {YELLOW} {!}
sigrity::update Trace {Trace03} -Net {Sig1} -Color {YELLOW} {!}
sigrity::update Trace {Trace01} -Net {Sig1} -Color {YELLOW} {!}
sigrity::update Trace {Trace011} -Net {Sig2} -Color {0x00007F} {!}
sigrity::update Trace {Trace012} -Net {Sig2} -Color {0x00007F} {!}
sigrity::update Trace {Trace013} -Net {Sig2} -Color {0x00007F} {!}
sigrity::update Trace {Trace014} -Net {Sig2} -Color {0x00007F} {!}
sigrity::update Trace {Trace015} -Net {Sig2} -Color {0x00007F} {!}
sigrity::update Trace {Trace016} -Net {Sig2} -Color {0x00007F} {!}
sigrity::update Trace {Trace017} -Net {Sig2} -Color {0x00007F} {!}
sigrity::update Trace {Trace020} -Net {Sig2} -Color {0x00007F} {!}
sigrity::update Trace {Trace019} -Net {Sig2} -Color {0x00007F} {!}
sigrity::update Trace {Trace018} -Net {Sig2} -Color {0x00007F} {!}
sigrity::hook -port {Port1} -DownVertical {!}
sigrity::hook -port {Port2} -DownVertical {!}
sigrity::hook -port {Port4} -DownVertical {!}
sigrity::hook -port {Port3} -DownVertical {!}
sigrity::save {!}
sigrity::open window freq {!}
sigrity::update freq -freq {0.000000, 1000000000.000000, 625000.000000, linear, 3} {1000000000.000000, 15000000000.000000, 17500000.000000, linear, 3} -start 0.000000 -end 15000000000.000000 -AFS -customize {!}
sigrity::begin simulation {!}
sigrity::update CurveView -source {S} -method {Amplitude} -XLog {0} -YLog {1} {!}
sigrity::save curve -netWork {SIMULATION} -fileName {D:\A0607\PowerSI\CFP4s4p\CFP4.s4p} -curveFileType {TouchStone} -matrixTypeToSave {S} -matrixDataType {RI} -freqUnit {GHZ}
sigrity::save {!}
sigrity::close document {!}