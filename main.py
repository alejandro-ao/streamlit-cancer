import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def clean_data(df):
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def plot_data(df):
    plot = df['diagnosis'].value_counts().plot(kind='bar', title="Class distributions \n(0: Benign | 1: Malignant)")
    plot.set_xlabel("Diagnosis")
    plot.set_ylabel("Frequency")
    plt.show()


def get_model():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    df = pd.read_csv('./data/cancer-diagnosis.csv')
    df = clean_data(df)

    # scale predictors and split data
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model
    y_pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    return model


def create_app(data):
    import streamlit as st

    # load the model
    model = get_model()

    # create the app
    st.title('Breast Cancer Detection')
    st.subheader('This app predicts if a cell cluster is benign or malignant')

    st.write("Enter the details of the cell nuclei in a digital image of a fine needle aspirate (FNA) of a breast mass to predict whether it is malignant or benign.")

    # Add a form to input the features
    radius = st.slider(
        "Radius (mean)", float(data.radius_mean.min()), float(data.radius_mean.max()),
        float(data.radius_mean.mean())
        )
    texture = st.slider(
        "Texture (mean)", float(data.texture_mean.min()), float(data.texture_mean.max()),
        float(data.texture_mean.mean())
        )
    perimeter = st.slider(
        "Perimeter (mean)", float(data.perimeter_mean.min()), float(data.perimeter_mean.max()),
        float(data.perimeter_mean.mean())
        )
    area = st.slider(
        "Area (mean)", float(data.area_mean.min()), float(data.area_mean.max()),
        float(data.area_mean.mean())
        )
    smoothness = st.slider(
        "Smoothness (mean)", float(data.smoothness_mean.min()), float(data.smoothness_mean.max()),
        float(data.smoothness_mean.mean())
        )
    compactness = st.slider(
        "Compactness (mean)", float(data.compactness_mean.min()),
        float(data.compactness_mean.max()), float(data.compactness_mean.mean())
        )
    concavity = st.slider(
        "Concavity (mean)", float(data.concavity_mean.min()), float(data.concavity_mean.max()),
        float(data.concavity_mean.mean())
        )
    concave_points = st.slider(
        "Concave points (mean)", float(data['concave points_mean'].min()),
        float(data['concave points_mean'].max()), float(data['concave points_mean'].mean())
        )
    symmetry = st.slider(
        "Symmetry (mean)", float(data.symmetry_mean.min()), float(data.symmetry_mean.max()),
        float(data.symmetry_mean.mean())
        )
    fractal_dimension = st.slider(
        "Fractal dimension (mean)", float(data.fractal_dimension_mean.min()),
        float(data.fractal_dimension_mean.max()), float(data.fractal_dimension_mean.mean())
        )
    radius_se = st.slider(
        "Radius (se)", float(data.radius_se.min()), float(data.radius_se.max()), float(data.radius_se.mean())
        )
    texture_se = st.slider(
        "Texture (se)", float(data.texture_se.min()), float(data.texture_se.max()), float(data.texture_se.mean())
        )
    perimeter_se = st.slider(
        "Perimeter (se)", float(data.perimeter_se.min()), float(data.perimeter_se.max()),
        float(data.perimeter_se.mean())
        )
    area_se = st.slider("Area (se)", float(data.area_se.min()), float(data.area_se.max()), float(data.area_se.mean()))
    smoothness_se = st.slider(
        "Smoothness (se)", float(data.smoothness_se.min()), float(data.smoothness_se.max()),
        float(data.smoothness_se.mean())
        )
    compactness_se = st.slider(
        "Compactness (se)", float(data.compactness_se.min()), float(data.compactness_se.max()),
        float(data.compactness_se.mean())
        )
    concavity_se = st.slider(
        "Concavity (se)", float(data.concavity_se.min()), float(data.concavity_se.max()),
        float(data.concavity_se.mean())
        )
    concave_points_se = st.slider(
        "Concave points (se)", float(data['concave points_se'].min()), float(data['concave points_se'].max()),
        float(data['concave points_se'].mean())
        )
    symmetry_se = st.slider(
        "Symmetry (se)", float(data.symmetry_se.min()), float(data.symmetry_se.max()), float(data.symmetry_se.mean())
        )
    fractal_dimension_se = st.slider(
        "Fractal dimension (se)", float(data.fractal_dimension_se.min()), float(data.fractal_dimension_se.max()),
        float(data.fractal_dimension_se.mean())
        )
    radius_worst = st.slider(
        "Radius (worst)", float(data.radius_worst.min()), float(data.radius_worst.max()),
        float(data.radius_worst.mean())
        )
    texture_worst = st.slider(
        "Texture (worst)", float(data.texture_worst.min()), float(data.texture_worst.max()),
        float(data.texture_worst.mean())
        )
    perimeter_worst = st.slider(
        "Perimeter (worst)", float(data.perimeter_worst.min()), float(data.perimeter_worst.max()),
        float(data.perimeter_worst.mean())
        )
    area_worst = st.slider(
        "Area (worst)", float(data.area_worst.min()), float(data.area_worst.max()), float(data.area_worst.mean())
        )
    smoothness_worst = st.slider(
        "Smoothness (worst)", float(data.smoothness_worst.min()), float(data.smoothness_worst.max()),
        float(data.smoothness_worst.mean())
        )
    compactness_worst = st.slider(
        "Compactness (worst)", float(data.compactness_worst.min()), float(data.compactness_worst.max()),
        float(data.compactness_worst.mean())
        )
    concavity_worst = st.slider(
        "Concavity (worst)", float(data.concavity_worst.min()), float(data.concavity_worst.max()),
        float(data.concavity_worst.mean())
        )
    concave_points_worst = st.slider(
        "Concave points (worst)", float(data['concave points_worst'].min()), float(data['concave points_worst'].max()),
        float(data['concave points_worst'].mean())
        )
    symmetry_worst = st.slider(
        "Symmetry (worst)", float(data.symmetry_worst.min()), float(data.symmetry_worst.max()),
        float(data.symmetry_worst.mean())
        )
    fractal_dimension_worst = st.slider(
        "Fractal dimension (worst)", float(data.fractal_dimension_worst.min()),
        float(data.fractal_dimension_worst.max()), float(data.fractal_dimension_worst.mean())
        )

    # scale the input
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    input = scaler.fit_transform(
        [[radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points,
            symmetry, fractal_dimension, radius_se, texture_se, perimeter_se, area_se, smoothness_se,
            compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst,
            texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
            concave_points_worst, symmetry_worst, fractal_dimension_worst]]
        )

    # Use the model to make a prediction
    prediction = model.predict(input)

    if prediction[0] == 0:
        st.write("The cell cluster is benign")
    else:
        st.write("The cell cluster is malignant")


def main():
    # EDA
    # df = pd.read_csv('./data/cancer-diagnosis.csv')
    # df = clean_data(df)
    # plot_data(df)

    # MODEL
    # model = get_model()
    # print("Model: ", model)

    # APP
    data = pd.read_csv('./data/cancer-diagnosis.csv')
    data = clean_data(data)
    create_app(data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
