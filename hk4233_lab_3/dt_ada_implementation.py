"""
@author : Hima Bindu Krovvidi
CSCI 630 Lab 3
"""

import math
import pickle
import random
import pandas as pd

class nodes:

    def __init__(self, index, name):
        """
        Constructor for the node class
        :param index: index of the node
        :param name: name of the node
        """
        self.index = index
        self.name = name
        self.left = None
        self.right = None
        self.weight = None

class decs_tree:
    """
    This function is used to predict the language of the given test data.
    """
    def __init__(self):
        """
        This function is used to initialize the variables.
        """
        self.english_dict = {"is", "for", "to", "in", "on", "at", "then", "than", "which", "whether", "and", "all", "did", "while",
        "might", "if", "the", "an", "of", "it", "was"}
        self.dutch_cons = {'ä', 'ö', 'ü', "ï", "ë"}
        self.dutch_dict = {"zij", "goed", "wat", "is", "alstublieft", "niet", "nog", "nou", "ja", "nee", "wel", "de", "het", "een", "van", "en", "dat", "ik", "je"}
        self.dutch_numb = {'een', "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen"}
        self.english_numb = {'one', "two", "three", "four", "five", "six", "seven", "eight", "nine"}
        self.vis = set()

    def generate_dt(self, inFile, outFile):
        """
        This function is used to generate the decision tree.
        """
        level = 1
        max_level = 5
        print("Fitting based on Training data.")
        frame = self.selection(inFile)
        print("Train decision tree.")
        visited_nodes = set()
        root_tree = self.create_decs_tree(frame, visited_nodes, level, max_level, [])
        print("Storing tree in a pickle file format.")
        curr = open(outFile, 'wb')
        pickle.dump(['dt', root_tree], curr)
        print("Tree storage has been completed.")

    def generate_ada(self, inFile, outFile):
        """
        This function is used to generate the adaboost.
        """
        print("Fitting based on Training data.")
        frame = self.selection(inFile)
        print("Train adaboost.")
        decis_tree = self.adaboost(None, frame, [])
        print("Storing Tree in a file in pickle format.")
        curr = open(outFile, 'wb')
        pickle.dump(['ada', decis_tree], curr)
        print("Tree storage completed.")

    def adaboost_pred(self, fname, feature):
        """
        This function is used to predict the language of the given test data.
        """
        decis_stub = fname
        adboost_output = []
        for _, i in feature.iterrows():
            list_of_attr = list(i)
            en_weight = 0
            nl_weight = 0
            for each in range(len(decis_stub)):
                """
                This loop is used to find the weight of the decision tree.
                """
                curr_ans = (self.labels_find(
                    decis_stub[each][0], list_of_attr))
                if curr_ans == 'en':
                    en_weight += decis_stub[each][1]
                else:
                    nl_weight += decis_stub[each][1]
            if en_weight > nl_weight:
                """
                This condition is used to find the language of the given test data.
                """
                adboost_output.append('en')
            else:
                adboost_output.append('nl')
        return adboost_output

    def test_to_fit(self, list_attr):
        """
        This function is used to test the data.
        """
        frame_pd = pd.DataFrame()
        for each in list_attr:
            """
            This loop is used to find the attributes of the data.
            """
            min_length = False
            avg_word_size = False
            i_or_j = False
            contains_q = False
            contains_dutch = False
            contains_english = False
            contains_number_12 = False
            contains_z = False
            contains_unique_dutch = False
            dutch_words_count = 0
            english_words_count = 0
            z_count = 0
            q_count = 0
            rows = each.split(" ")
            contains_feat = rows
            words_count = len(contains_feat)
            sum_total = 0
            for attr in contains_feat:
                if attr in self.dutch_dict:
                    """
                    This condition is used to find the dutch words.
                    """
                    dutch_words_count += 1
                if len(attr) > 1:
                    """
                    This condition is used to find the length of the word.
                    """
                    if attr[0] == 'z':
                        z_count += 1
                if attr in self.english_dict:
                    """
                    This condition is used to find the english words.
                    """
                    english_words_count += 1
                if len(attr) >= 11:
                    """
                    This condition is used to find the minimum length of the word.
                    """
                    contains_number_12 = True
                if attr.isalpha():
                    if len(attr) == 1:
                        min_length = True
                    else:
                        """
                        This condition is used to find the average word size.
                        """
                        for i in range(len(attr) - 1):
                            if attr[i] in self.dutch_cons:
                                contains_unique_dutch = True
                            if attr[i] == 'i' and attr[i + 1] == 'j':
                                i_or_j = True
                            if attr[i] == 'Q' or attr[i] == 'q':
                                q_count += 1
                                if q_count > 1:
                                    contains_q = True
                sum_total += len(attr)
            average_count = sum_total / words_count
            if average_count > 5:
                avg_word_size = True
            if dutch_words_count > 2:
                contains_dutch = True
            if english_words_count > 1 and dutch_words_count > 1:
                if english_words_count > dutch_words_count:
                    contains_dutch = False
                else:
                    contains_dutch = True
            if z_count >= 2:
                contains_z = True
            if english_words_count >= 4:
                contains_english = True
            each = pd.DataFrame({"C1": [min_length], "C2": [avg_word_size], "C3": [i_or_j],
                                 "C4": [contains_q],
                                 "C5": [contains_dutch], "C6": [contains_number_12], "C7": [contains_z],
                                 "C8": [contains_unique_dutch],
                                 "C9": [contains_english]})
            frame_pd = pd.concat([frame_pd, pd.DataFrame(each)])

        return frame_pd

    def selection(self, examples):
        """
        This function is used to select the data.
        """
        frames = open(examples, encoding="utf8")
        list_of_dataframe = []
        for curr_frame in frames:
            minimum_length = False
            average_word_size = False
            i_or_j = False
            contains_q = False
            contains_dutch = False
            contains_english = False
            contains_number_12 = False
            contains_letter_z = False
            contains_unique_dutch_words = False
            dutch_words_count = 0
            english_words_count = 0
            z_count = 0
            q_count = 0
            rows = curr_frame.split('|')
            final_node = rows[0].strip()
            contains_attr = rows[1].split(" ")
            words_count = len(contains_attr)
            sum_total = 0
            for each in contains_attr:
                if each in self.dutch_dict:
                    dutch_words_count += 1
                if len(each) > 1:
                    if each[0] == 'z':
                        z_count += 1
                if (len(each) >= 11):
                    contains_number_12 = True
                if each in self.english_dict:
                    english_words_count += 1
                if each.isalpha():
                    for i in range(len(each) - 1):
                        if (each[i] in self.dutch_cons):
                            contains_unique_dutch_words = True
                        if each[i] == 'i' and each[i + 1] == 'j':
                            i_or_j = True
                        if each[i] == 'Q' or each[i] == 'q':
                            q_count += 1
                        if q_count > 1:
                            contains_q = True
                if each.isalpha() and len(each) == 1:
                    """
                    This function checks if the word is alpha and length is 1 so that it can be considered as minimum length.
                    """
                    minimum_length = True
                sum_total = sum_total + len(each)

            average = sum_total / words_count
            if average > 5:
                """
                This condition checks if the average word size is greater than 5.
                """
                average_word_size = True
            if dutch_words_count > 2:
                """
                This condition checks if the dutch words count is greater than 2.
                """
                contains_dutch = True
            if english_words_count > 1 and dutch_words_count > 1:
                """
                This condition checks if the english words count is greater than 1 and dutch words count is greater than 1.
                """
                if english_words_count > dutch_words_count:
                    contains_dutch = False
                else:
                    contains_dutch = True
            if english_words_count >= 4:
                """
                This condition checks if the english words count is greater than or equal to 4.
                """
                contains_english = True
            if z_count >= 2:
                """
                This condition checks if the z count is greater than or equal to 2.
                """
                contains_letter_z = True
            if final_node.strip() == "en":
                """
                This condition checks if the final node is equal to en.
                """
                final_node = 'en'
            else:
                final_node = 'nl'
            curr_frame = pd.DataFrame({"C1": [minimum_length], "C2": [average_word_size], "C3": [i_or_j],
                                 "C4": [contains_q],
                                 "C5": [contains_dutch], "C6": [contains_number_12], "C7": [contains_letter_z],
                                 "C8": [contains_unique_dutch_words],
                                 "C9": [contains_english], "destination": [final_node]})
            list_of_dataframe.append(curr_frame)
        frame_data = pd.concat(list_of_dataframe)

        return frame_data

    def generate_name(self, frame):
        """
        This function is used to generate the name of the file.
        """
        positives = 0
        negatives = 0
        final_class = list(frame)
        for each in range(len(final_class)):
            """
            This loop is used to count the number of positives and negatives.
            """
            if final_class[each] == 'en':
                positives += 1
            elif final_class[each] == 'nl':
                negatives += 1
            if negatives > positives:
                label = "nl"
            else:
                label = "en"
        return label

    vis = set()
    level = 1
    max_level = 4

    def label_find(self, curr, tree):
        """
        This function is used to find the label of the node.
        """
        output_lis = []
        for _, each in curr.iterrows():
            """
            This loop is used to iterate over the rows of the dataframe.
            """
            x = list(each)
            output_lis.append(self.labels_find(tree, x))
        return output_lis

    def labels_find(self, tree, value):
        """
        This function is used to find the labels of the node.
        """
        if not tree.left and not tree.right:
            """
            This condition checks if the left and right of the tree is None.
            """
            return tree.name
        if value[int(tree.index[1]) - 1]:
            """
            This condition checks if the value of the index is true.
            """
            temp = self.labels_find(tree.left, value)
            return temp
        else:
            """
            This condition checks if the value of the index is false.
            """
            temp = self.labels_find(tree.right, value)
            return temp

    def calc_entropy(self, frame, column, final_node, weight):
        """
        This function is used to calculate the entropy.
        The probabilities are used to calculate the entropy.
        """
        a1_probability = 0
        a2_probability = 0
        a3_probability = 0
        a4_probability = 0
        final_class = list(frame.iloc[:, -1])
        positives = 0
        negatives = 0
        final_node = list(final_node)
        for each in range(len(final_node)):
            if final_class[each] == 'en':
                positives += 1
            elif final_class[each] == 'nl':
                negatives += 1
        lis = list(column)
        for each in range(len(lis)):
            """
            This loop is used to calculate the probabilities.
            """
            if lis[each] == True and final_node[each] == 'en':
                a1_probability += 1
            elif lis[each] == False and final_node[each] == 'en':
                a2_probability += 1
            elif lis[each] == True and final_node[each] == 'nl':
                a3_probability += 1
            elif lis[each] == False and final_node[each] == 'nl':
                a4_probability += 1
        True_entropy = 0
        False_entropy = 0
        if a1_probability != 0 and a3_probability != 0:
            """
            This condition checks if the a1 probability is not equal to 0 and a3 probability is not equal to 0.
            """
            True_entropy = (-a1_probability / (a1_probability + a3_probability) *
                            math.log(a1_probability / (a1_probability + a3_probability), 2)) - (
                                   a3_probability / (a3_probability + a1_probability) *
                                   math.log(a3_probability / (a1_probability + a3_probability), 2))
        if a2_probability != 0 and a4_probability != 0:
            """
            This condition checks if the a2 probability is not equal to 0 and a4 probability is not equal to 0.
            """
            False_entropy = (-a2_probability / (a2_probability + a4_probability) *
                             math.log(a2_probability / (a2_probability + a4_probability), 2)) - (
                                    a4_probability / (a4_probability + a2_probability) *
                                    math.log(a4_probability / (a2_probability + a4_probability), 2))
        total_entropy = ((a1_probability + a3_probability) / (positives + negatives) * True_entropy) + (
                (a2_probability + a4_probability) / (positives + negatives) * False_entropy)
        return total_entropy

    def adaboost(self, tree, frame, weights):
        """
        This function is used to implement the adaboost algorithm.
        """
        final_attrs = []
        length_of_frame = len(frame)
        for each in range(len(frame)):
            weights.append(1 / len(frame))
        compare_ans = list(frame.iloc[:, -1])
        total_stubs = []
        new_tree = None
        frame1 = frame
        frame_2 = frame
        error_total = []
        vis_set = set()
        level = 1
        max_level = 4
        for each in range(10):
            """
            This loop is used to iterate over the number of stumps.
            """
            frame1 = frame_2
            matrix_adb = frame1
            matrix_pred = matrix_adb.iloc[:, :-1]
            new_tree = self.create_decs_tree(
                matrix_adb, vis_set, level, max_level, weights)
            final_attrs = (self.label_find(matrix_pred, new_tree))
            total_error = 0.0
            for each in range(length_of_frame):
                if compare_ans[each] != final_attrs[each]:
                    total_error = total_error + weights[each]

            error_total.append(total_error)
            curr_error = (1 - total_error) / total_error
            norm_weights = 0.5 * (math.log(curr_error))

            for each in range(length_of_frame):
                """
                This loop is used to iterate over the length of the frame.
                """
                if final_attrs[each] == compare_ans[each]:
                    weights[each] = weights[each] * (math.e ** (-norm_weights))
                else:
                    weights[each] = weights[each] * (math.e ** norm_weights)
            length_of_weights = len(weights)
            for each in range(length_of_weights):
                """
                This loop is used to iterate over the length of the weights.
                """
                weights[each] = weights[each] / math.fsum(weights)

            total_stubs.append((new_tree, norm_weights))
            while len(frame_2) < len(frame):
                total_range = random.uniform(0, 1)
                for each in range(len(weights) - 1):
                    if weights[each] < weights[each + 1]:
                        min_weight = weights[each]
                        max_weight = weights[each + 1]
                    else:
                        min_weight = weights[each + 1]
                        max_weight = weights[each]
                    if min_weight < total_range < max_weight:
                        new_value = frame1.iloc[[each]]
                        frame_2 = frame_2.append(new_value)
                    if len(frame_2) >= len(frame):
                        break
        return total_stubs

    def create_decs_tree(self, frame1, vis_set, level, max_level, weight):
        """
        This function is used to create the decision tree.
        """
        true_nodes = None
        false_nodes = None
        english_probability = 0
        dutch_probability = 0
        positives = 0
        negatives = 0
        if frame1 is None:
            return
        store_class = list(frame1.iloc[:, -1])
        for each in range(len(store_class)):
            """
            This loop is used to iterate over the length of the store class.
            """
            if store_class[each] == 'nl':
                negatives += 1
            elif store_class[each] == 'en':
                positives += 1
        if negatives != 0 and positives != 0:
            """
            This condition checks if the negatives are not equal to 0 and positives are not equal to 0.
            """
            english_probability = positives / (negatives + positives)
            dutch_probability = negatives / (negatives + positives)
            """
            This condition checks if the english probability is greater than the dutch probability.
            """
        if negatives == 0 and positives != 0:
            return nodes('en', 'en')
        elif positives == 0 and negatives != 0:
            return nodes('nl', 'nl')
        if english_probability > 0.96:
            return nodes('en', 'en')
        elif dutch_probability > 0.96:
            return nodes('nl', 'nl')
        lis_of_lis = frame1.iloc[:, :-1]
        final_node = frame1.iloc[:, -1]
        entropy = []
        lowest = float('inf')
        for each in lis_of_lis:
            if each not in self.vis:
                entropy.append(self.calc_entropy(
                    frame1, lis_of_lis[each], final_node, weight))
            if each in self.vis:
                entropy.append(1)
        for each in range(len(entropy)):
            """
            This loop is used to iterate over the length of the entropy.
            """
            if entropy[each] < lowest:
                lowest = entropy[each]
                min_index = frame1.columns[each]
                """
                This condition checks if the entropy is less than the lowest.
                """
        name = self.generate_name(final_node)
        root = nodes(min_index, name)
        self.vis.add(min_index)
        if level >= max_level:
            return root
        level += 1
        if lowest == 0:
            return root
        subnode = list(set(frame1[min_index]))
        """
        This condition checks if the lowest is equal to 0.
        """
        if len(subnode) >= 2:
            false_nodes = frame1[frame1[min_index] == subnode[0]]
            true_nodes = frame1[frame1[min_index] == subnode[1]]
        root.left = self.create_decs_tree(
            true_nodes, self.vis, level, max_level, weight)
        root.right = self.create_decs_tree(
            false_nodes, self.vis, level, max_level, weight)
        return root
