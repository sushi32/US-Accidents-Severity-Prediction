import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.model_selection import cross_val_score


def LR_KNN_Model():
    data = pd.read_pickle("../data/processed_data.pkl")
    df = data.sample(n=1000000, random_state=108)

    df.drop(columns=['Start_Time', 'End_Time', 'Weather_Timestamp', 'Description', 'Street', 'City', 'County', 'State',
                     'Side', 'Zipcode', 'Timezone', 'Airport_Code', 'Wind_Direction', 'Weather_Condition',
                     'Sunrise_Sunset',
                     'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Start_Year', 'Start_Month',
                     'End_Lat', 'End_Lng',
                     'Duration(min)', 'Distance(mi)', 'Start_Lat', 'Start_Lng'], inplace=True)

    y = df['Severity']
    X = df.drop('Severity', axis=1)

    bestfeatures = SelectKBest(k=7)
    fit = bestfeatures.fit(X, y)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    print(featureScores.nlargest(15, 'Score'))

    X = df[
        ['Traffic_Signal', 'Start_Hour', 'Crossing', 'Temperature(F)', 'Wind_Chill(F)', 'Wind_Speed(mph)', 'Junction',
         'Humidity(%)',
         'Pressure(in)', 'Visibility(mi)', 'Start_Weekday']]
    y = df['Severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    '''
    -------------------------------------LR-------------------------------------------------------------------
    '''
    model = LogisticRegression(random_state=42, multi_class='multinomial')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    count_misclassified = (y_test != y_pred).sum()
    print('Misclassifications: {}'.format(count_misclassified))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracy)
    titles = [("Confusion matrix without normalization", None),
              ("Normalized confusion matrix", 'true')]
    for title, normalize in titles:
        disp = plot_confusion_matrix(model, X_test, y_test,
                                     normalize=normalize, cmap="YlGnBu")
        disp.ax_.set_title(title)

        plt.grid(False)
        print(title)
        print(disp.confusion_matrix)

    plt.savefig("../plots/logisticRegression_Confusion_matrix_1.png")
    plt.close()

    sm = SMOTE(sampling_strategy={3: 150000, 4: 125000, 1: 75000}, random_state=67)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(Counter(y_train))
    rus = RandomUnderSampler({2: 200000}, random_state=67)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    print(Counter(y_train))

    model = LogisticRegression(random_state=42, multi_class='multinomial')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    count_misclassified = (y_test != y_pred).sum()
    print('Misclassifications: {}'.format(count_misclassified))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracy)

    titles = [("Confusion matrix without normalization", None),
              ("Normalized confusion matrix", 'true')]
    for title, normalize in titles:
        disp = plot_confusion_matrix(model, X_test, y_test,
                                     normalize=normalize, cmap="YlGnBu")
        disp.ax_.set_title(title)

        plt.grid(False)
        print(title)
        print(disp.confusion_matrix)

    plt.savefig("../plots/logisticRegression_Confusion_matrix_2.png")
    plt.close()

    '''
    -------------------------------------KNN-------------------------------------------------------------------
    '''

    test_error = []
    train_accuracy = []
    test_accuracy = []
    for i in range(1, 20, 2):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        test_error.append(np.mean(pred_i != y_test))
        train_accuracy.append(knn.score(X_train, y_train))
        test_accuracy.append(knn.score(X_test, y_test))

    plt.figure(figsize=(12, 6))
    epoch_count = range(1, 21, 2)
    test_error = [0.315345, 0.3842, 0.407755, 0.4215, 0.423135, 0.42405, 0.425415, 0.424305, 0.42343, 0.42261]
    plt.plot(epoch_count, test_error)
    plt.xticks(epoch_count, epoch_count)
    plt.xlabel("K")
    plt.ylabel("Error")
    plt.savefig("../plots/kNN_test_error_plot.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    epoch_count = range(1, 21, 2)
    plt.plot(epoch_count, train_accuracy, label="Train accuracy")
    plt.plot(epoch_count, test_accuracy, label="Test accuracy")
    plt.xticks(epoch_count, epoch_count)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("../plots/kNN_train_test_accuracy_plot.png")
    plt.close()

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print('[K-Nearest Neighbors (KNN), n_neighbors = 10] knn.score: {:.3f}'.format(knn.score(X_test, y_test)))
    print('[K-Nearest Neighbors (KNN), n_neighbors = 10] accuracy_score: {:.3f}'.format(acc))

    scores_cv = []
    for i in range(1, 11):
        scores_k = []
        knn = KNeighborsClassifier(n_neighbors=i)
        for j in range(2, 6):
            cv = cross_val_score(knn, X, y, cv=j)
            scores_k.append(cv.tolist())
        scores_cv.append(scores_k)

        mean_scores_cv = []
    for i in range(len(scores_cv)):
        lst = []
        for j in range(len(scores_cv[i])):
            val = sum(scores_cv[i][j]) / len(scores_cv[i][j])
            lst.append(val)
        mean_scores_cv.append(lst)
    print(mean_scores_cv)

    plt.figure(figsize=(20, 8))
    cv_epoch = range(2, 6)
    for i in range(1, 11):
        l = "K: " + str(i)
        plt.plot(cv_epoch, mean_scores_cv[i - 1], label=l)
    plt.xticks(cv_epoch, cv_epoch)
    plt.xlabel("CV")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("../plots/kNN_cv_accuracy_plot.png")
    plt.close()

    scores_cv2 = cross_val_score(knn, X, y, cv=2)
    print(np.mean(scores_cv2))
    print(round(np.var(scores_cv2), 1))

    scores_cv3 = cross_val_score(knn, X, y, cv=3)
    print(np.mean(scores_cv3))
    print(round(np.var(scores_cv3), 1))


if __name__ == "__main__":
    LR_KNN_Model()