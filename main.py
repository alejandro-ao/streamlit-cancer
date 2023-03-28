import pandas as pd
import matplotlib.pyplot as plt

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


def create_radar_chart(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                       concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                       radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
                       concave_points_se, symmetry_se, fractal_dimension_se,
                       radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst,
                       concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst):

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean,
               concave_points_mean, symmetry_mean, fractal_dimension_mean],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Mean'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
               concave_points_se, symmetry_se, fractal_dimension_se],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Standard Error'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst,
               concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Worst'
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 60]
            )
        ),
        showlegend=True,
        autosize=False,
        width=400,
        height=700,
    )

    return fig


def create_input_form(data):
    import streamlit as st

    st.sidebar.header("Cell Nuclei Details")
    # Add a form to input the features
    radius = st.sidebar.slider(
        "Radius (mean)", float(data.radius_mean.min()), float(data.radius_mean.max()),
        float(data.radius_mean.mean())
    )
    texture = st.sidebar.slider(
        "Texture (mean)", float(data.texture_mean.min()), float(data.texture_mean.max()),
        float(data.texture_mean.mean())
    )
    perimeter = st.sidebar.slider(
        "Perimeter (mean)", float(data.perimeter_mean.min()), float(data.perimeter_mean.max()),
        float(data.perimeter_mean.mean())
    )
    area = st.sidebar.slider(
        "Area (mean)", float(data.area_mean.min()), float(data.area_mean.max()),
        float(data.area_mean.mean())
    )
    smoothness = st.sidebar.slider(
        "Smoothness (mean)", float(data.smoothness_mean.min()), float(data.smoothness_mean.max()),
        float(data.smoothness_mean.mean())
    )
    compactness = st.sidebar.slider(
        "Compactness (mean)", float(data.compactness_mean.min()),
        float(data.compactness_mean.max()), float(data.compactness_mean.mean())
    )
    concavity = st.sidebar.slider(
        "Concavity (mean)", float(data.concavity_mean.min()), float(data.concavity_mean.max()),
        float(data.concavity_mean.mean())
    )
    concave_points = st.sidebar.slider(
        "Concave points (mean)", float(data['concave points_mean'].min()),
        float(data['concave points_mean'].max()), float(data['concave points_mean'].mean())
    )
    symmetry = st.sidebar.slider(
        "Symmetry (mean)", float(data.symmetry_mean.min()), float(data.symmetry_mean.max()),
        float(data.symmetry_mean.mean())
    )
    fractal_dimension = st.sidebar.slider(
        "Fractal dimension (mean)", float(data.fractal_dimension_mean.min()),
        float(data.fractal_dimension_mean.max()), float(data.fractal_dimension_mean.mean())
    )
    radius_se = st.sidebar.slider(
        "Radius (se)", float(data.radius_se.min()), float(data.radius_se.max()), float(data.radius_se.mean())
    )
    texture_se = st.sidebar.slider(
        "Texture (se)", float(data.texture_se.min()), float(data.texture_se.max()), float(data.texture_se.mean())
    )
    perimeter_se = st.sidebar.slider(
        "Perimeter (se)", float(data.perimeter_se.min()), float(data.perimeter_se.max()),
        float(data.perimeter_se.mean())
    )
    area_se = st.sidebar.slider(
        "Area (se)", float(data.area_se.min()), float(data.area_se.max()), float(data.area_se.mean())
        )
    smoothness_se = st.sidebar.slider(
        "Smoothness (se)", float(data.smoothness_se.min()), float(data.smoothness_se.max()),
        float(data.smoothness_se.mean())
    )
    compactness_se = st.sidebar.slider(
        "Compactness (se)", float(data.compactness_se.min()), float(data.compactness_se.max()),
        float(data.compactness_se.mean())
    )
    concavity_se = st.sidebar.slider(
        "Concavity (se)", float(data.concavity_se.min()), float(data.concavity_se.max()),
        float(data.concavity_se.mean())
    )
    concave_points_se = st.sidebar.slider(
        "Concave points (se)", float(data['concave points_se'].min()), float(data['concave points_se'].max()),
        float(data['concave points_se'].mean())
    )
    symmetry_se = st.sidebar.slider(
        "Symmetry (se)", float(data.symmetry_se.min()), float(data.symmetry_se.max()), float(data.symmetry_se.mean())
    )
    fractal_dimension_se = st.sidebar.slider(
        "Fractal dimension (se)", float(data.fractal_dimension_se.min()), float(data.fractal_dimension_se.max()),
        float(data.fractal_dimension_se.mean())
    )
    radius_worst = st.sidebar.slider(
        "Radius (worst)", float(data.radius_worst.min()), float(data.radius_worst.max()),
        float(data.radius_worst.mean())
    )
    texture_worst = st.sidebar.slider(
        "Texture (worst)", float(data.texture_worst.min()), float(data.texture_worst.max()),
        float(data.texture_worst.mean())
    )
    perimeter_worst = st.sidebar.slider(
        "Perimeter (worst)", float(data.perimeter_worst.min()), float(data.perimeter_worst.max()),
        float(data.perimeter_worst.mean())
    )
    area_worst = st.sidebar.slider(
        "Area (worst)", float(data.area_worst.min()), float(data.area_worst.max()), float(data.area_worst.mean())
    )
    smoothness_worst = st.sidebar.slider(
        "Smoothness (worst)", float(data.smoothness_worst.min()), float(data.smoothness_worst.max()),
        float(data.smoothness_worst.mean())
    )
    compactness_worst = st.sidebar.slider(
        "Compactness (worst)", float(data.compactness_worst.min()), float(data.compactness_worst.max()),
        float(data.compactness_worst.mean())
    )
    concavity_worst = st.sidebar.slider(
        "Concavity (worst)", float(data.concavity_worst.min()), float(data.concavity_worst.max()),
        float(data.concavity_worst.mean())
    )
    concave_points_worst = st.sidebar.slider(
        "Concave points (worst)", float(data['concave points_worst'].min()), float(data['concave points_worst'].max()),
        float(data['concave points_worst'].mean())
    )
    symmetry_worst = st.sidebar.slider(
        "Symmetry (worst)", float(data.symmetry_worst.min()), float(data.symmetry_worst.max()),
        float(data.symmetry_worst.mean())
    )
    fractal_dimension_worst = st.sidebar.slider(
        "Fractal dimension (worst)", float(data.fractal_dimension_worst.min()),
        float(data.fractal_dimension_worst.max()), float(data.fractal_dimension_worst.mean())
    )

    input_data = {
        'radius_mean': radius,
        'texture_mean': texture,
        'perimeter_mean': perimeter,
        'area_mean': area,
        'smoothness_mean': smoothness,
        'compactness_mean': compactness,
        'concavity_mean': concavity,
        'concave points_mean': concave_points,
        'symmetry_mean': symmetry,
        'fractal_dimension_mean': fractal_dimension,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst
    }

    return input_data


def create_app(data):
    import streamlit as st

    # load the model
    model = get_model()

    # create the app
    st.set_page_config(page_title="Breast Cancer Diagnosis", page_icon=":female-doctor:", layout="wide", initial_sidebar_state="expanded")
    st.title("Breast Cancer Diagnosis")
    # create the input form
    input_data = create_input_form(data)


    radius = input_data['radius_mean']
    texture = input_data['texture_mean']
    perimeter = input_data['perimeter_mean']
    area = input_data['area_mean']
    smoothness = input_data['smoothness_mean']
    compactness = input_data['compactness_mean']
    concavity = input_data['concavity_mean']
    concave_points = input_data['concave points_mean']
    symmetry = input_data['symmetry_mean']
    fractal_dimension = input_data['fractal_dimension_mean']

    radius_se = input_data['radius_se']
    texture_se = input_data['texture_se']
    perimeter_se = input_data['perimeter_se']
    area_se = input_data['area_se']
    smoothness_se = input_data['smoothness_se']
    compactness_se = input_data['compactness_se']
    concavity_se = input_data['concavity_se']
    concave_points_se = input_data['concave points_se']
    symmetry_se = input_data['symmetry_se']
    fractal_dimension_se = input_data['fractal_dimension_se']

    radius_worst = input_data['radius_worst']
    texture_worst = input_data['texture_worst']
    perimeter_worst = input_data['perimeter_worst']
    area_worst = input_data['area_worst']
    smoothness_worst = input_data['smoothness_worst']
    compactness_worst = input_data['compactness_worst']
    concavity_worst = input_data['concavity_worst']
    concave_points_worst = input_data['concave points_worst']
    symmetry_worst = input_data['symmetry_worst']
    fractal_dimension_worst = input_data['fractal_dimension_worst']


    # scale the input
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    input_data = scaler.fit_transform(
        [[radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points,
            symmetry, fractal_dimension, radius_se, texture_se, perimeter_se, area_se, smoothness_se,
            compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst,
            texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
            concave_points_worst, symmetry_worst, fractal_dimension_worst]]
        )

    # Use the model to make a prediction
    prediction = model.predict(input_data)

    col1, col2 = st.columns([4,1])

    with col1:
        #st.write("Radar chart of the input data")
        radar_chart = create_radar_chart(
            radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points,
            symmetry, fractal_dimension, radius_se, texture_se, perimeter_se, area_se, smoothness_se,
            compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst,
            texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
            concave_points_worst, symmetry_worst, fractal_dimension_worst
            )
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        st.subheader('This app predicts if a cell cluster is benign or malignant')

        st.write(
            "Enter the details of the cell nuclei in a digital image of a fine needle aspirate (FNA) of a breast mass to predict whether it is malignant or benign."
            )
        st.write("The cell cluster is: ")
        if prediction[0] == 0:
            st.write("Benign")
        else:
            st.write("Malignant")

        st.write("Probability of being benign: ", model.predict_proba(input_data)[0][0])
        st.write("Probability of being malignant: ", model.predict_proba(input_data)[0][1])


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
