import sys
import dt_ada_implementation

def dt_training(data, fname):
    dt = dt_ada_implementation.decs_tree()
    print("Training Decision Tree")
    dt.generate_dt(data, fname)

def ada_training(data, fname):
    dt = dt_ada_implementation.decs_tree()
    print("Training AdaBoost")
    dt.generate_ada(data, fname)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Please enter args:")
        print("python train.py <training-set> <to-print> <model-file>")
        exit(0)
    else:
        data = sys.argv[1]
        to_print = sys.argv[2]
        type = sys.argv[3]
        if type == "dt":
            dt_training(sys.argv[1], sys.argv[2])
        elif type == "ada":
            ada_training(sys.argv[1], sys.argv[2])
        else:
            print("Please enter args:")
            print("python train.py <train_file> <model_file> <dt/ada>")