import numpy as np
import neurolab as nl

# B A O
target = [
    [1, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 0],

    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1],

    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0]
]

chars = ['B', 'A', 'O']
target = np.asfarray(target)
target[target == 0] = -1

# Create and train network
net = nl.net.newhop(target)

output = net.sim(target)
print("Test on train samples:")
for i in range(len(output)):
    print(chars[i], np.array_equal(np.sign(output[i]), target[i]))

def test_defaced(letter_index, test_pattern):
    print(f"Test of defaced {chars[letter_index]}:")
    test_pattern[test_pattern == 0] = -1
    output = net.sim([test_pattern])
    print(np.array_equal(np.sign(output[0]), target[letter_index]), 'Sim. steps', len(net.layers[0].outs))

# Тесты для "B", "A" и "O"
test_defaced(0, [1, 1, 1, 1, 0,
                 1, 0, 0, 0, 1,
                 1, 1, 1, 1, 0,
                 1, 0, 0, 0, 1,
                 1, 1, 1, 1, 0])

test_defaced(1, [0, 1, 1, 1, 0,
                 1, 0, 0, 0, 1,
                 1, 1, 1, 1, 1,
                 1, 0, 0, 0, 1,
                 1, 0, 0, 0, 1])

test_defaced(2, [0, 1, 1, 1, 0,
                 1, 0, 0, 0, 1,
                 1, 0, 0, 0, 1,
                 1, 0, 0, 0, 1,
                 0, 1, 1, 1, 0])
