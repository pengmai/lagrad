func @mrelatives_to_absolutes(%relatives: tensor<22x4x4xf64>, %parents: tensor<22xi32>) -> tensor<22x4x4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n_one = arith.constant -1 : i32
  %cb = arith.constant 22 : index
  %matmul_init = arith.constant dense<0.0> : tensor<4x4xf64>
  %absolute_space = linalg.init_tensor [22, 4, 4] : tensor<22x4x4xf64>
  %absolutes = scf.for %iv = %c0 to %cb step %c1 iter_args(%a_iter = %absolute_space) -> (tensor<22x4x4xf64>) {
    %parent_i = tensor.extract %parents[%iv] : tensor<22xi32>
    %rel_i = tensor.extract_slice %relatives[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
    %pred = arith.cmpi "eq", %parent_i, %n_one : i32
    %result = scf.if %pred -> tensor<4x4xf64> {
      scf.yield %rel_i : tensor<4x4xf64>
    } else {
      %parent_idx = arith.index_cast %parent_i : i32 to index
      %abs_p = tensor.extract_slice %a_iter[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
      // This is the ADBench orientation, not the Enzyme orientation.
      %abs_i = linalg.matmul ins(%abs_p, %rel_i : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) -> tensor<4x4xf64>
      scf.yield %abs_i : tensor<4x4xf64>
    }
    %a_next = tensor.insert_slice %result into %a_iter[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<22x4x4xf64>
    scf.yield %a_next : tensor<22x4x4xf64>
  }
  return %absolutes : tensor<22x4x4xf64>
}

// #matmul_adjoint_arg0 = {
//   indexing_maps = [
//     affine_map<(d0, d1, d2) -> (d0, d1)>,
//     affine_map<(d0, d1, d2) -> (d2, d1)>,
//     affine_map<(d0, d1, d2) -> (d0, d2)>
//   ],
//   iterator_types = ["parallel", "reduction", "parallel"]
// }
// #matmul_adjoint_arg1 = {
//   indexing_maps = [
//     affine_map<(d0, d1, d2) -> (d1, d0)>,
//     affine_map<(d0, d1, d2) -> (d1, d2)>,
//     affine_map<(d0, d1, d2) -> (d0, d2)>
//   ],
//   iterator_types = ["parallel", "reduction", "parallel"]
// }

// func @mygrad_rels_to_abs(%relatives: tensor<22x4x4xf64>, %parents: tensor<22xi32>) -> tensor<22x4x4xf64> {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c3 = arith.constant 3 : index
//   %n_one = arith.constant -1 : i32
//   %cb = arith.constant 22 : index
//   %matmul_init = arith.constant dense<0.0> : tensor<4x4xf64>
//   %absolute_space = linalg.init_tensor [22, 4, 4] : tensor<22x4x4xf64>
//   %absolutes = scf.for %iv = %c0 to %cb step %c1 iter_args(%a_iter = %absolute_space) -> (tensor<22x4x4xf64>) {
//     %parent_i = tensor.extract %parents[%iv] : tensor<22xi32>
//     %rel_i = tensor.extract_slice %relatives[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
//     %pred = arith.cmpi "eq", %parent_i, %n_one : i32
//     %result = scf.if %pred -> tensor<4x4xf64> {
//       scf.yield %rel_i : tensor<4x4xf64>
//     } else {
//       %parent_idx = arith.index_cast %parent_i : i32 to index
//       %abs_p = tensor.extract_slice %a_iter[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
//       // This is the ADBench orientation, not the Enzyme orientation.
//       %abs_i = linalg.matmul ins(%abs_p, %rel_i : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) -> tensor<4x4xf64>
//       scf.yield %abs_i : tensor<4x4xf64>
//     }
//     %a_next = tensor.insert_slice %result into %a_iter[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<22x4x4xf64>
//     scf.yield %a_next : tensor<22x4x4xf64>
//   }

//   %one = arith.constant 1.0 : f64
//   %zero = arith.constant 0.0 : f64
//   %last = arith.constant 21 : index
//   %dabsolutes_space = linalg.init_tensor [22, 4, 4] : tensor<22x4x4xf64>
//   %dabsolutes_init = linalg.fill(%one, %dabsolutes_space) : f64, tensor<22x4x4xf64> -> tensor<22x4x4xf64>
//   %drelatives_space = linalg.init_tensor [22, 4, 4] : tensor<22x4x4xf64>
//   %drelatives_init = linalg.fill(%zero, %drelatives_space) : f64, tensor<22x4x4xf64> -> tensor<22x4x4xf64>
//   %dres:2 = scf.for %iv = %c0 to %cb step %c1 iter_args(%dabs = %dabsolutes_init, %drel = %drelatives_init) -> (tensor<22x4x4xf64>, tensor<22x4x4xf64>) {
//     %idx = arith.subi %last, %iv : index
//     %da_next = tensor.extract_slice %dabs[%idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
//     %parent_i = tensor.extract %parents[%idx] : tensor<22xi32>
//     %rel_i = tensor.extract_slice %relatives[%idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
//     %pred = arith.cmpi "eq", %parent_i, %n_one : i32
//     %dresult:2 = scf.if %pred -> (tensor<4x4xf64>, tensor<22x4x4xf64>) {
//       scf.yield %da_next, %dabs : tensor<4x4xf64>, tensor<22x4x4xf64>
//     } else {
//       %parent_idx = arith.index_cast %parent_i : i32 to index
//       %abs_p = tensor.extract_slice %absolutes[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
//       %dabs_i_wrt_abs_p = linalg.generic #matmul_adjoint_arg0 ins(%da_next, %rel_i : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) {
//       ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
//         %0 = arith.mulf %arg0, %arg1 : f64
//         %1 = arith.addf %0, %arg2 : f64
//         linalg.yield %1 :  f64
//       } -> tensor<4x4xf64>
//       %dabs_i_wrt_rel_i = linalg.generic #matmul_adjoint_arg1 ins(%abs_p, %da_next : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) {
//       ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
//         %0 = arith.mulf %arg0, %arg1 : f64
//         %1 = arith.addf %0, %arg2 : f64
//         linalg.yield %1 :  f64
//       } -> tensor<4x4xf64>

//       %dabs_next_0 = tensor.extract_slice %dabs[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
//       %dabs_next_1 = arith.addf %dabs_next_0, %dabs_i_wrt_abs_p : tensor<4x4xf64>
//       %dabs_next = tensor.insert_slice %dabs_next_1 into %dabs[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<22x4x4xf64>
//       scf.yield %dabs_i_wrt_rel_i, %dabs_next : tensor<4x4xf64>, tensor<22x4x4xf64>
//     }
//     // This is mainly for efficiency reasons
//     %drel_slice_0 = tensor.extract_slice %drel[%idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
//     %drel_slice_1 = arith.addf %drel_slice_0, %dresult#0 : tensor<4x4xf64>
//     %drel_next = tensor.insert_slice %drel_slice_1 into %drel[%idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<22x4x4xf64>
//     scf.yield %dresult#1, %drel_next : tensor<22x4x4xf64>, tensor<22x4x4xf64>
//   }
//   return %dres#1 : tensor<22x4x4xf64>
// }

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %relatives = arith.constant dense<[
    [[0.9205236434936523, -0.1684776246547699, 0.3524934947490692, 0.0],[-0.11152654886245728, -0.9780153036117554, -0.1762043535709381, 0.0], [0.374430388212204, 0.12288789451122284, -0.9190759062767029, 0.0], [0.0, 0.0, 0.0, 1.0]],
    [[0.6583698987960815, 0.5094536542892456, -0.5540811419487, 7.450580430390374e-11], [0.548811674118042, -0.8286978006362915, -0.1098436638712883, 0.003628205507993698], [-0.5151260495185852, -0.2317684143781662, -0.8251838088035583, 0.0], [0.0, 0.0, 0.0, 1.0]],
    [[-0.13738234848852215, 0.986556862951591, -0.08849630828955318, -1.9371508841459217e-09], [-0.897033359455838, -0.08602911276736691, 0.4335089205195475, 0.035285357385873795], [0.4200679856098901, 0.13894062720664824, 0.8967934823758674, -8.940696516468449e-10], [0.0, 0.0, 0.0, 1.0]],
    [[0.9768162369728088, -0.04013700040423096, -0.21028307858311557, -1.19209286886246e-09], [0.14163677394390106, 0.8577189581344298, 0.49422343458580453, 0.038056597113609314], [0.16052713990211487, -0.5125493449099894, 0.8435188574099337, 9.834765890559538e-09], [0.0, 0.0, 0.0, 1.0]],
    [[0.9968672394752502, -0.015383403198196755, 0.07758261986052983, -1.7881393032936899e-09], [0.07155107706785202, 0.5934289267148843, -0.8016996833629385, 0.04199063032865524], [-0.03370688855648041, 0.8047392967458533, 0.5926706653321672, -4.4703482582342247e-10], [0.0, 0.0, 0.0, 1.0]],
    [[0.9156719446182251, 0.14837013185024261, 0.37353822588920593, 7.450580430390374e-11], [0.10956025123596191, -0.9863159656524658, 0.12319641560316086, 0.003628205507993698], [0.3867056369781494, -0.07188259065151215, -0.919397234916687, 0.0], [0.0, 0.0, 0.0, 1.0]],
    [[0.8152171903278749, -0.5355925933628847, 0.2203667300149264, -1.6477132991354893e-09], [0.5791172568282681, 0.7494822836037689, -0.32077962055933407, 0.10898979008197784], [0.006646240685994342, 0.38912321754990203, 0.9211617119300859, 2.9103830456733704e-11], [0.0, 0.0, 0.0, 1.0]],
    [[0.9990891814231873, 0.04155726022837012, 0.009685334836412263, 1.883599942686942e-09], [-0.042410995811223984, 0.9921027588242705, 0.11804037141700102, 0.03234409913420677], [-0.0047034090384840965, -0.11834360333735083, 0.9929614625536184, -7.683410851999639e-11], [0.0, 0.0, 0.0, 1.0]],
    [[0.999750018119812, 0.006315097417979451, -0.021447961234848073, -1.5731529967588642e-10], [-0.022355183959007263, 0.2985113854732675, -0.9541443576169353, 0.028275057673454285], [0.0003769324393942952, 0.9543852315632002, 0.2985779145665447, -5.587935322792781e-11], [0.0, 0.0, 0.0, 1.0]],
    [[0.9260880351066589, -0.04630468413233757, 0.37445563077926636, 7.450580430390374e-11], [-0.09915236383676529, -0.9874266982078552, 0.12311563640832901, 0.003628205507993698], [0.36404672265052795, -0.15114407241344452, -0.9190350770950317, 0.0], [0.0, 0.0, 0.0, 1.0]],
    [[0.7620278237027565, 0.6472757981555085, -0.018647996920962917, -4.95634817909707e-11], [-0.6468841706123794, 0.762235229775904, 0.02320307182655082, 0.1073712632060051], [0.029232965109829867, -0.005618277821785672, 0.9995568724354061, 3.811464685532506e-11], [0.0, 0.0, 0.0, 1.0]],
    [[0.9995884895324707, 0.021399873970132155, -0.019102575167493932, 5.555921278599385e-10], [-0.028665713965892792, 0.7699768523461845, -0.6374275262855975, 0.03411087766289711], [0.0010676837991923094, 0.6377128175485784, 0.7702734588498616, 2.421438605182402e-10], [0.0, 0.0, 0.0, 1.0]],
    [[0.9989896416664124, -0.007923458313694703, -0.044238500022131255, -3.813702242894124e-09], [0.04494118317961693, 0.18358099486460477, 0.9819767973841901, 0.029941830784082413], [0.00034067645901814103, -0.9829727145331767, 0.18375159108557781, 2.9103828722010228e-12], [0.0, 0.0, 0.0, 1.0]],
    [[0.8889668583869934, -0.267085462808609, 0.37202566862106323, 7.450580430390374e-11], [-0.3302887976169586, -0.9366239309310913, 0.11681258678436279, 0.003628205507993698], [0.31724902987480164, -0.2267184853553772, -0.9208427667617798, 0.0], [0.0, 0.0, 0.0, 1.0]],
    [[0.9997074023993386, -0.018558212845700917, -0.015518473478364103, 1.224725598714116e-10], [0.023917208616112795, 0.8545497567397854, 0.5188184864209012, 0.10859718918800354], [0.0036329591553673074, -0.5190378016752788, 0.8547435551379874, 1.1641531488804091e-11], [0.0, 0.0, 0.0, 1.0]],
    [[0.9994715452194214, -0.026219811084212896, -0.019211765282234783, 7.332709994756215e-10], [0.032450638711452484, 0.7707112234293343, 0.6363578573186486, 0.02825337089598179], [-0.001878466922789812, -0.6366449080369514, 0.7711546734337191, -1.8626450382086546e-10], [0.0, 0.0, 0.0, 1.0]],
    [[0.9999964833259583, -9.671638310312924e-05, -0.0026578737245938067, 2.439483060001635e-09], [0.0022619804367423058, 0.5565780032135423, 0.8307923966483948, 0.022544240579009056], [0.0013989637373015285, -0.8307952782751649, 0.5565761129303448, 1.490116086078075e-10], [0.0, 0.0, 0.0, 1.0]],
    [[0.8128832578659058, -0.4039934277534485, 0.41953548789024353, 7.450580430390374e-11], [-0.5053189992904663, -0.8473770022392273, 0.16310986876487732, 0.003628205507993698], [0.2896096706390381, -0.3445884585380554, -0.8929643034934998, 0.0], [0.0, 0.0, 0.0, 1.0]],
    [[-0.902921826407237, -0.40109469497686845, 0.1544504973960492, 1.6950070236276815e-09], [0.4286919299514693, -0.8145870501320897, 0.3907317514419164, 0.1046757847070694], [-0.030907064888159752, 0.4190119594853551, 0.9074546457914314, 7.450580430390374e-11], [0.0, 0.0, 0.0, 1.0]],
    [[0.9981788992881775, 0.031791684799135675, 0.05126814107104084, -4.091416538898329e-09], [-0.06015810742974281, 0.5877886954661076, 0.8067748138520532, 0.02187417633831501], [-0.004486088640987873, -0.8083896729515239, 0.5886307050610036, -5.21540644005114e-10], [0.0, 0.0, 0.0, 1.0]],
    [[0.9978939294815063, 0.05801491793788216, -0.029017475370930912, -4.7916546286330686e-09], [-0.06404736638069153, 0.9520929974226462, -0.29902626147036226, 0.017919939011335373], [0.010279330424964428, 0.3002549611232892, 0.9538035067015969, -1.19209286886246e-09], [0.0, 0.0, 0.0, 1.0]],
    [[-0.9527995586395264, 0.09495650231838226, 0.2883681356906891, 7.450580430390374e-11], [0.1363903284072876, 0.9824634790420532, 0.12713387608528137, 0.003628205507993698], [-0.2712389826774597, 0.16046370565891266, -0.949042022228241, 0.0], [0.0, 0.0, 0.0, 1.0]]
  ]> : tensor<22x4x4xf64>
  %parents = arith.constant dense<[-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19, 0]> : tensor<22xi32>
  %f = constant @mrelatives_to_absolutes : (tensor<22x4x4xf64>, tensor<22xi32>) -> tensor<22x4x4xf64>
  %df = standalone.grad %f {of = [0]} : (tensor<22x4x4xf64>, tensor<22xi32>) -> tensor<22x4x4xf64>, (tensor<22x4x4xf64>, tensor<22xi32>) -> tensor<22x4x4xf64>
  %res = call_indirect %df(%relatives, %parents) : (tensor<22x4x4xf64>, tensor<22xi32>) -> tensor<22x4x4xf64>
  // %res = call @mygrad_rels_to_abs(%relatives, %parents) : (tensor<22x4x4xf64>, tensor<22xi32>) -> tensor<22x4x4xf64>
//   // %res = call @mrelatives_to_absolutes(%relatives, %parents) : (tensor<22x4x4xf64>, tensor<22xi32>) -> tensor<22x4x4xf64>
  %U = tensor.cast %res : tensor<22x4x4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}