import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Load dataset
column_names = [
    'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon',
    'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement',
    'scalp_involvement', 'family_history', 'melanin_incontinence', 'eosinophils_in_the_infiltrate',
    'PNL_infiltrate', 'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis',
    'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges',
    'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule', 'munro_microabcess',
    'focal_hypergranulosis', 'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer',
    'spongiosis', 'saw_tooth_appearance_of_retes', 'follicular_horn_plug', 'perifollicular_parakeratosis',
    'inflammatory_monoluclear_inflitrate', 'band_like_infiltrate', 'age', 'class'
]

df = pd.read_csv("dermatology.data", header=None, names=column_names, na_values="?")
df.dropna(inplace=True)
df['class'] = df['class'].astype(int)

# Train model
X = df.drop("class", axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

# Home Page
if page == "Home":
    st.title("🏠 DermaScan")
    st.write("Welcome to **DermaScan**, a skin disease classifier app using the UCI Dermatology Dataset.")
    st.markdown("- Use the **Predict** section to get a disease class based on symptoms.")
    st.markdown("- View **Dataset**, **Summary**, and **Graphs** to understand the data.")

# Dataset Page
elif page == "Dataset":
    st.title("📊 Dataset Viewer")
    st.dataframe(df)

# Summary Page
elif page == "Summary":
    st.title("📄 Dataset Summary")
    st.write(df.describe())

# Graphs Page
elif page == "Graphs":
    st.title("📈 Visual Analysis")
    st.bar_chart(df['class'].value_counts())

# Predict Page
elif page == "Predict":
    st.title("🧠 Predict Skin Disease Class")
    st.info("Adjust the sliders below based on patient symptoms:")

    user_input = []
    for feature in X.columns:
        val = st.slider(f"{feature}", int(df[feature].min()), int(df[feature].max()), int(df[feature].mean()))
        user_input.append(val)

    if st.button("Predict"):
        prediction = clf.predict([user_input])[0]
        st.success(f"🎯 Predicted Disease Class: {prediction}")
        acc = accuracy_score(y_test, clf.predict(X_test))
        st.write(f"📊 Model Accuracy: **{acc:.2f}**")
