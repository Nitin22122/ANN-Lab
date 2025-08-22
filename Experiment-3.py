#Implement the perceptron model for basic logical functions like AND, OR, NOT, NAND, NOR.


def perceptron_gate_threshold(gate, inputs):
    n = len(inputs)

    if gate == "AND":
        weights = [1] * n
        threshold = n

    elif gate == "OR":
        weights = [1] * n
        threshold = 1

    elif gate == "NAND":
        weights = [1] * n
        threshold = n

    elif gate == "NOR":
        weights = [1] * n
        threshold = 1

    elif gate == "NOT":
        if n != 1:
            return "X NOT gate only accepts 1 input."
        weights = [-1]
        threshold = 0

    else:
        return "X Invalid gate selected."

    summation = sum(w * x for w, x in zip(weights, inputs))

    if gate == "NAND" or gate == "NOR":
        output = 0 if summation >= threshold else 1
    else:
        output = 1 if summation >= threshold else 0

    return output


while True:
    print("\nAvailable Gates: AND, OR, NOT, NAND, NOR")
    gate = input("Enter logic gate: ").strip().upper()

    try:
        num_inputs = int(input("\nEnter number of inputs: "))
        if gate == "NOT" and num_inputs != 1:
            print("X NOT gate only accepts 1 input")
            continue
    except ValueError:
        print("X Invalid number!")
        continue

    inputs = []
    for i in range(num_inputs):
        while True:
            try:
                x = int(input(f"Enter input {i+1} (0 or 1): "))
                if x not in [0, 1]:
                    print("X only 0 or 1 allowed.")
                    continue
                inputs.append(x)
                break
            except ValueError:
                print("X Enter valid integer (0 or 1).")

    result = perceptron_gate_threshold(gate, inputs)
    print(f"Gate: {gate}, Inputs: {inputs} => {result}")

    again = input("\nDo you want to try again? (yes/no): ").strip().lower()
    if again != "yes":
        print("Program exited.")
        break
