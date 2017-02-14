from progressbar import ProgressBar


def pad_string(string, length):
    return (string + spaces(length-len(string)))[:length]


def spaces(n):
    string = ''
    for i in range(0, n):
        string += ' '
    return string


def count_correct_characters(data, predictions, max_length):
    nr_characters = len(data) * max_length
    nr_spaces = 0
    nr_correct = 0
    nr_correct_spaces = 0
    with ProgressBar(max_value=len(data)) as progressbar:
        for i in range(len(data)):
            data_string = pad_string(data[i].decode('utf-8'), max_length)
            predicted_string = pad_string(predictions[i].decode('utf-8'), max_length)
            for j in range(max_length):
                if data_string[j] is ' ':
                    nr_spaces += 1
                if data_string[j] is predicted_string[j]:
                    nr_correct += 1
                    if data_string[j] is ' ':
                        nr_correct_spaces += 1
            progressbar.update(i)
    return nr_characters, nr_correct, nr_spaces, nr_correct_spaces


def count_by_character(data, predictions, max_length):
    nr_in_data = {}
    nr_in_predictions = {}
    nr_correct = {}
    with ProgressBar(max_value=len(data)) as progressbar:
        for i in range(len(data)):
            data_string = pad_string(data[i].decode('utf-8'), max_length)
            predicted_string = pad_string(predictions[i].decode('utf-8'), max_length)
            for j in range(max_length):
                data_char = data_string[j]
                predicted_char = predicted_string[j]
                if data_char in nr_in_data:
                    nr_in_data[data_char] += 1
                else:
                    nr_in_data[data_char] = 1
                if predicted_char in nr_in_predictions:
                    nr_in_predictions[predicted_char] += 1
                else:
                    nr_in_predictions[predicted_char] = 1
                if data_char is predicted_char:
                    if data_char in nr_correct:
                        nr_correct[data_char] += 1
                    else:
                        nr_correct[data_char] = 1
            progressbar.update(i)
    return nr_in_data, nr_in_predictions, nr_correct
