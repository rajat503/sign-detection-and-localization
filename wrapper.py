def train_classifier(train_list, path):
    import read_classify_top5
    read_classify_top5.train(train_list, path)

def train_localizer(train_list, path):
    import read_localization
    read_localization.train(train_list, path)

def test_classifier(test_image, x1, y1, x2, y2):
    import read_classify_top5
    return read_classify_top5.test(test_image, x1, y1, x2, y2)

def test_localizer(test_image):
    import localization
    coordinates = localization.test(test_image)
    return coordinates[0][0], coordinates[1][0], coordinates[2][0], coordinates[3][0]
