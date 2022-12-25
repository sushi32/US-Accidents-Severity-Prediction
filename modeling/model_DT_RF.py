import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
from imblearn.over_sampling import SMOTE
from collections import Counter


def Dt_Rf_Model():
    df = pd.read_pickle("../data/processed_data.pkl")
    df["Weekend"] = np.where(((df['Start_Weekday'] == 0) | (df['Start_Weekday'] == 6)), True, False)
    d = dict(Day=0, Night=1)
    df['Sunrise_Sunset'] = df['Sunrise_Sunset'].map(d)
    df['Civil_Twilight'] = df['Civil_Twilight'].map(d)
    df['Nautical_Twilight'] = df['Nautical_Twilight'].map(d)
    df['Astronomical_Twilight'] = df['Astronomical_Twilight'].map(d)
    booleanCols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                   'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
                   'Turning_Loop', 'Weekend']
    df[booleanCols] = (df[booleanCols] == True).astype(int)
    df.head()
    ca_df1 = df[df['State'] == 'CA']
    ca_df1.drop(
        columns=['Start_Time', 'End_Time', 'Weather_Timestamp', 'Description', 'Street', 'City', 'County', 'State',
                 'Side', 'Zipcode', 'Timezone', 'Airport_Code', 'Wind_Direction', 'Civil_Twilight', 'Nautical_Twilight',
                 'Astronomical_Twilight', 'End_Lat', 'End_Lng', 'Duration(min)', 'Distance(mi)'], inplace=True)
    ca_X = ca_df1.drop(['Severity', 'Weather_Condition'], axis=1)
    ca_y = ca_df1['Severity']
    X_train_ca, X_test_ca, y_train_ca, y_test_ca = train_test_split(ca_X.values, ca_y.values, random_state=43)

    """
    ---------------------------------DT 1----------------------------------------------------------------
    """
    clf = DecisionTreeClassifier()
    clf.fit(X_train_ca, y_train_ca)
    pred_ca = clf.predict(X_test_ca)

    print("Accuracy = {:.3f}".format(accuracy_score(y_test_ca, pred_ca)))

    print("F1 score:", f1_score(y_test_ca, pred_ca, average='weighted'))

    print("Recall score:", recall_score(y_test_ca, pred_ca, average='weighted'))

    print("My prediction: {}".format(clf.predict(X_test_ca)[0:25]))
    print("Actual result: {}".format(y_test_ca[0:25]))

    matrix_ca = confusion_matrix(y_test_ca, pred_ca)
    acc_per_class_ca = matrix_ca.diagonal() / matrix_ca.sum(axis=0)

    disp = plot_confusion_matrix(clf, X_test_ca, y_test_ca, normalize="true", cmap="YlGnBu")
    disp.ax_.set_title("Confusion Matrix")
    plt.grid(False)
    plt.savefig("../plots/decisionTree_Confusion_matrix_1.png")
    plt.close()
    print('Decision Tree 1 Accuracy per class:', acc_per_class_ca)

    '''
    -----------------------RF 1-------------------------------------------------------------------------
    '''
    rf = RandomForestClassifier(n_estimators=50, max_features='auto', random_state=3)
    rf.fit(X_train_ca, y_train_ca)

    pred_rfca = rf.predict(X_test_ca)

    print(f"accuracy = {rf.score(X_test_ca, y_test_ca)}")

    print("F1 score:", f1_score(y_test_ca, pred_rfca, average='weighted'))

    print("Recall score:", recall_score(y_test_ca, pred_rfca, average='weighted'))

    matrix_rfca = confusion_matrix(y_test_ca, pred_rfca)
    acc_per_class_rfca = matrix_rfca.diagonal() / matrix_rfca.sum(axis=0)

    disp = plot_confusion_matrix(rf, X_test_ca, y_test_ca, normalize="true", cmap="YlGnBu")
    disp.ax_.set_title("Confusion Matrix")
    plt.grid(False)
    plt.savefig("../plots/randomForest_Confusion_matrix_1.png")
    plt.close()

    print('Random FOrest 1 Accuracy per class:', acc_per_class_rfca)

    good_weather = list({'Clear', 'Fair', 'Fair / Windy'})
    mild_weather = list(
        {'Blowing Dust', 'Partly Cloudy', 'Scattered Clouds', 'Mostly Cloudy', 'Light Drizzle', 'Light Hail',
         'Overcast', 'Mist', 'Smoke', 'Smoke / Windy', 'Shallow Fog', 'Cloudy / Windy', 'Drizzle', 'Small Hail',
         'Cloudy', 'Mostly Cloudy / Windy', 'N/A Precipitation', 'Partial Fog', 'Light Rain Shower', 'Light Snow',
         'Partly Cloudy', 'Partly Cloudy / Windy', 'Patches of Fog', 'Showers in the Vicinity', 'Light Ice Pellets'})
    bad_weather = []
    lst = good_weather + mild_weather
    weather_vals_set = set(ca_df1['Weather_Condition'])
    for x in weather_vals_set:
        if x not in lst:
            bad_weather.append(x)

    def simplify_vals(col: str, value: list, replacement: list, df=ca_df1):
        df[col] = df[col].replace(value, replacement * len(value))

    simplify_vals(col='Weather_Condition', value=good_weather, replacement=[0])
    simplify_vals(col='Weather_Condition', value=mild_weather, replacement=[1])
    simplify_vals(col='Weather_Condition', value=bad_weather, replacement=[2])

    '''
        -----------------------DT 2-------------------------------------------------------------------------
    '''
    ca_X = ca_df1.drop(['Severity'], axis=1)
    ca_y = ca_df1['Severity']
    X_train_ca, X_test_ca, y_train_ca, y_test_ca = train_test_split(ca_X.values, ca_y.values, random_state=43)

    clf = DecisionTreeClassifier()
    clf.fit(X_train_ca, y_train_ca)
    pred_ca = clf.predict(X_test_ca)

    print("Accuracy = {:.3f}".format(accuracy_score(y_test_ca, pred_ca)))

    print("F1 score:", f1_score(y_test_ca, pred_ca, average='weighted'))

    print("Recall score:", recall_score(y_test_ca, pred_ca, average='weighted'))

    matrix_ca = confusion_matrix(y_test_ca, pred_ca)
    acc_per_class_ca = matrix_ca.diagonal() / matrix_ca.sum(axis=0)

    disp = plot_confusion_matrix(clf, X_test_ca, y_test_ca, normalize="true", cmap="YlGnBu")
    disp.ax_.set_title("Confusion Matrix")
    plt.grid(False)
    plt.savefig("../plots/decisionTree_Confusion_matrix_2.png")
    plt.close()

    print('Accuracy per class:', acc_per_class_ca)

    print(classification_report(y_test_ca, pred_ca, digits=4))

    '''
        -----------------------RF 2-------------------------------------------------------------------------
    '''
    rf = RandomForestClassifier(n_estimators=50, max_features='auto', random_state=3)
    rf.fit(X_train_ca, y_train_ca)

    pred_rfca = rf.predict(X_test_ca)

    print(f"accuracy = {rf.score(X_test_ca, y_test_ca)}")

    print("F1 score:", f1_score(y_test_ca, pred_rfca, average='weighted'))

    print("Recall score:", recall_score(y_test_ca, pred_rfca, average='weighted'))

    matrix_rfca = confusion_matrix(y_test_ca, pred_rfca)
    acc_per_class_rfca = matrix_rfca.diagonal() / matrix_rfca.sum(axis=0)

    disp = plot_confusion_matrix(rf, X_test_ca, y_test_ca, normalize="true", cmap="YlGnBu")
    disp.ax_.set_title("Confusion Matrix")
    plt.grid(False)
    plt.savefig("../plots/randomForest_Confusion_matrix_2.png")
    plt.close()

    print('Accuracy per class:', acc_per_class_rfca)

    print(classification_report(y_test_ca, pred_rfca, digits=4))

    '''
        -----------------------DT 3-------------------------------------------------------------------------
    '''
    sm = SMOTE(random_state=137)
    print('Original dataset shape %s' % Counter(y_train_ca))
    X_train_ca, y_train_ca = sm.fit_resample(X_train_ca, y_train_ca)
    print('Resampled dataset shape %s' % Counter(y_train_ca))

    clf = DecisionTreeClassifier()
    clf.fit(X_train_ca, y_train_ca)
    pred_ca = clf.predict(X_test_ca)

    print("Accuracy = {:.3f}".format(accuracy_score(y_test_ca, pred_ca)))

    print("F1 score:", f1_score(y_test_ca, pred_ca, average='weighted'))

    print("Recall score:", recall_score(y_test_ca, pred_ca, average='weighted'))

    matrix_ca = confusion_matrix(y_test_ca, pred_ca)
    acc_per_class_ca = matrix_ca.diagonal() / matrix_ca.sum(axis=0)

    disp = plot_confusion_matrix(clf, X_test_ca, y_test_ca, normalize="true", cmap="YlGnBu")
    disp.ax_.set_title("Confusion Matrix")
    plt.grid(False)
    plt.savefig("../plots/decisionTree_Confusion_matrix_3.png")
    plt.close()

    print('Accuracy per class:', acc_per_class_ca)

    print(classification_report(y_test_ca, pred_ca, digits=4))

    '''
        -----------------------RF 3-------------------------------------------------------------------------
    '''
    rf = RandomForestClassifier(n_estimators=50, max_features='auto', random_state=3)
    rf.fit(X_train_ca, y_train_ca)

    pred_rfca = rf.predict(X_test_ca)

    print(f"accuracy = {rf.score(X_test_ca, y_test_ca)}")

    print("F1 score:", f1_score(y_test_ca, pred_rfca, average='weighted'))

    print("Recall score:", recall_score(y_test_ca, pred_rfca, average='weighted'))

    matrix_rfca = confusion_matrix(y_test_ca, pred_rfca)
    acc_per_class_rfca = matrix_rfca.diagonal() / matrix_rfca.sum(axis=0)

    disp = plot_confusion_matrix(rf, X_test_ca, y_test_ca, normalize="true", cmap="YlGnBu")
    disp.ax_.set_title("Confusion Matrix")
    plt.grid(False)
    plt.savefig("../plots/randomForest_Confusion_matrix_3.png")
    plt.close()

    print('Accuracy per class:', acc_per_class_rfca)

    print(classification_report(y_test_ca, pred_rfca, digits=4))

    tree.plot_tree(clf, max_depth=5)

    dot_data = tree.export_graphviz(clf, out_file=None, max_depth=6, feature_names=ca_X.columns, class_names='Severity',
                                    filled=True, rounded=True, special_characters=True)
    pydot_graph = pydotplus.graph_from_dot_data(dot_data)
    pydot_graph.write_png('original_tree.png')
    pydot_graph.set_size('"7,12!"')
    pydot_graph.write_png('../plots/resized_tree.png')


if __name__ == "__main__":
    Dt_Rf_Model()
