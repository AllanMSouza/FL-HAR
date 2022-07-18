import numpy as np
from csvec import CSVec
import torch

RO = 0.8 # Momentum rate
ETA = 1.0  # Learning rate

# CSVec parameters
CSVEC_C = 2000
CSVEC_COMPRESSION = 0.1
CSVEC_K = 300

def uncompressClientsGradient(gradient, s_u, s_e):
    global csvec

    if s_u is None:
        s_u = torch.zeros(csvec.table.shape)
        s_e = torch.zeros(csvec.table.shape)

    csvec.zero()

    # for client in clientscompressedgradientobject:
    csvec.accumulateTable(gradient)

    s_u = RO * s_u + csvec.table #/ 1 #len(clientscompressedgradientobject)

    csvec.table = ETA * s_u + s_e
    delta       = csvec.unSketch(k=CSVEC_K)

    csvec.zero()
    csvec.accumulateVec(delta)
    s_e = ETA * s_u + s_e - csvec.table

    return s_u, s_e, delta


def addGradient(modelNotTrained, delta):
    dimension_counter = 0
    new_weight = []
    delta      = delta.numpy()

    for weight in modelNotTrained:
        shape = weight.shape
        delta_slice = delta[dimension_counter: dimension_counter + np.prod(shape)].reshape(shape)
        new_weight.append(weight - delta_slice)
        dimension_counter += np.prod(shape)

    return new_weight
    #modelNotTrained.set_weights(new_weight)


def computeClientGradient(modelNotTrained, modelTrained):
  gradient = []
  notTrainedWeight = modelNotTrained
  i = 0
  for weight in modelTrained:
    gradient.append( notTrainedWeight[i] - weight )
    i += 1

  return gradient

def compressGradient(clientgradient):
    global csvec
    csvec = None

    print_compression = False

    gradient_vector = np.concatenate([g.reshape((-1,)) for g in clientgradient])
    gradients_size  = gradient_vector.shape[0]

    if csvec is None:
        print_compression = True
        csvec_rows = max(1, round(CSVEC_COMPRESSION*gradients_size/CSVEC_C))
        csvec = CSVec(gradients_size, CSVEC_C, csvec_rows)
        # print(f'Creating CSVEC Table\n   Columns: {CSVEC_C}\n   Rows: {csvec_rows}\n   '
        #       f'Compression: {CSVEC_C*csvec_rows/gradients_size}')
    csvec.zero()
    csvec.accumulateVec(gradient_vector)

    compressedclientgradient = csvec.table.numpy()
    compressedclientgradient = np.array(compressedclientgradient, dtype=np.float16)

    if print_compression:
        import sys
        # print(f"Original size (in bytes):   {sys.getsizeof(gradient_vector)}")
        # print(f"Compressed size (in bytes): {sys.getsizeof(compressedclientgradient)}")
        print(f"Model Compressed: {sys.getsizeof(compressedclientgradient)/sys.getsizeof(gradient_vector)}")

        # log_compression = open('Comprresion_log/CS.csv', 'a')
        # log_compression.write(f'CS, {sys.getsizeof(gradient_vector)}, {sys.getsizeof(compressedclientgradient)}, {sys.getsizeof(compressedclientgradient)/sys.getsizeof(gradient_vector)}\n')
        # log_compression.close()

    return compressedclientgradient