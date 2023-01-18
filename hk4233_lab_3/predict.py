import pickle
import dt_ada_implementation
import sys


def test_dt(file, fname):
    dt = dt_ada_implementation.decs_tree()
    print("Testing Decision Tree")
    frame = dt.test_to_fit(file)
    pickle_dt = open(fname, 'rb')
    type, root_node = pickle.load(pickle_dt)
    if (type == "dt"):
        final_res = (dt.label_find(frame, root_node))
    else:
        final_res = dt.adaboost_pred(root_node, frame)
    return final_res


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please enter args:")
        print("python predict.py <test-set> <model-file>")
        exit(0)
    else:
        file_add = sys.argv[1]
        fname = sys.argv[2]
        file_read = []
        with open(fname) as read_curr:
            for line in read_curr:
                file_read.append(line)
        res = test_dt(file_read, file_add)
        for each in res:
            print(each)
