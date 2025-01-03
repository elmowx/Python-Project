import pandas as pd
import streamlit as st
from PIL import Image
from model import model_prediction
import json
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def process_main_page():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Car Price Prediction",
        page_icon=Image.open('data/car.jpg'),
    )
    page = st.sidebar.selectbox("Select a page", ["Data Analysis", "Price Prediction"])

    if page == "Data Analysis":
        show_main_page()
    elif page == "Price Prediction":
        show_prediction_page()


def show_main_page():
    st.write(
        """
        # This is the data analysis page
        Use the sidebar to navigate to data input and predictions.
        """
    )

    st.write("### Original Dataset")
    df = pd.read_csv('data/data.csv')
    st.write(df.head(5))

    with st.expander("Data Description", expanded=False):
        st.markdown(
            """
            **Target Variable**
            - `selling_price`: selling price, numeric

            **Features**
            - **`name`** *(string)*: car model
            - **`year`** *(numeric, int)*: year of manufacture
            - **`km_driven`** *(numeric, int)*: mileage at the time of sale
            - **`fuel`** *(categorical: _Diesel_, _Petrol_, _CNG_, _LPG_, _electric_)*: fuel type
            - **`seller_type`** *(categorical: _Individual_, _Dealer_, _Trustmark Dealer_)*: seller type
            - **`transmission`** *(categorical: _Manual_, _Automatic_)*: transmission type
            - **`owner`** *(categorical: _First Owner_, _Second Owner_, _Third Owner_, _Fourth & Above Owner_)*: which owner is it?
            - **`mileage`** *(string, numerically interpreted)*: mileage, requires preprocessing
            - **`engine`** *(string, numerically interpreted)*: engine volume, requires preprocessing
            - **`max_power`** *(string, numerically interpreted)*: peak engine power, requires preprocessing
            - **`torque`** *(string, numerically interpreted, possibly two values)*: torque, requires preprocessing
            - **`seats`** *(numeric, float; logically categorical, int)*: number of seats
            """
        )

    with st.expander("Check for NaN", expanded=True):
        nan_summary = df.isnull().sum()
        if nan_summary.sum() == 0:
            st.success("No missing values in the data!")
        else:
            st.warning("There are missing values in the data!")
            st.write("Number of missing values per column:")
            st.write(nan_summary[nan_summary > 0])

    st.write("### Dataset after some transformations")
    st.write("name -> Company, Model; Mileage transform; Engine transform; Max_power transform;")
    df_v1 = pd.read_csv('data/data_preproc_v1.csv')
    st.write(df_v1.head(5))
    
    st.write("### Main numerical characteristics")
    st.write(df.describe())
    st.markdown(
        """
        **Describes key statistics for numerical columns.**
        - year: Ranges from 1983 to 2020, with a mean of 2013.82.
        - selling_price: Varies from 29,999 to 10 million, having a mean price of 639,512.
        - km_driven: Goes from 1,000 to about 2.36 million kilometers.
        - mileage: The mean is 19.52 km/l, with no recorded mileage for some entries (min = 0).
        - engine: Ranges from 624 cc to 3604 cc, averaging 1458.33 cc.
        - max_power: Extends from 0 to 400 (unknown unit), with an average of 91.5.
        - seats: Mostly between 2 and 14, with a mean of 5.42, although some data is missing.
        """
    )

    st.write("### Categorical Characteristics")
    st.write(df.describe(include='object'))
    st.markdown(
        """
        **Summarizes categorical data.**
        - Company: 30 unique car companies; most frequent is Maruti (2126 entries).
        - Model: 206 distinct models, with "Swift" being the most common (852 entries).
        - fuel: 4 types; Diesel is the most used.
        - seller_type: 3 types, mainly Individual sellers (5186 entries).
        - transmission: Primarily Manual (5924 entries), with 2 types.
        - owner: Typically First Owner (4809 entries).
        - torque: 419 unique values; most frequent is "190Nm@2000rpm" (468 entries).
        """
    )

    with st.expander("EDA", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Correlation Matrix of Numerical Features")
            corr_matrix = df_v1.select_dtypes(include=['int64', 'float64']).corr()
            f, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(
                corr_matrix,
                annot=True,
                linewidths=0.5,
                linecolor="red",
                fmt=".4f",
                cmap="coolwarm",
                ax=ax
            )
            st.pyplot(f)
            st.write(
                "The correlation matrix illustrates the relationships between different" 
                "numerical features such as year, selling price, km driven, mileage, engine, max power, and seats."
                "From the chart, it is clear that `engine` and `max_power` are most correlated, "
                "which makes sense since peak engine power is directly dependent on engine volume."
            )

            st.write("### Dependence of selling price on mileage")
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='km_driven', y='selling_price', data=df_v1)
            plt.title('Dependence of Selling Price on Mileage')
            plt.xlabel('Mileage (km)')
            plt.ylabel('Selling Price (selling_price)')
            st.pyplot(plt)
            st.write(
                "The scatter plot illustrates the relationship between the sale price of cars and their mileage (kilometers driven)." 
                "The chart indicates that cars with lower mileage generally maintain higher sale prices, while higher mileage"
                "corresponds to a wider spread in pricing but tends to lower values. The plot underscores the negative correlation" 
                "between mileage and sale price, highlighting how usage impacts car value."
            )

            st.write("### Fuel Type Distribution")
            types_fuel = df_v1["fuel"].unique()
            x_fuel = df_v1["fuel"].value_counts()
            f, ax = plt.subplots(figsize=(5, 5))
            ax.pie(x_fuel, labels=types_fuel, autopct="%1.1f%%", startangle=90)
            ax.set_title("Fuel Type Distribution")
            st.pyplot(f)
            st.write(
                "The pie chart illustrates the distribution of fuel types in the dataset." 
                "Diesel is the most prevalent, accounting for 54.2% of the vehicles."
                "Petrol follows at 44.6%. CNG and LPG are much less common, each making up a small fraction of the total at 0.5% and 0.7%, respectively." 
                "This chart highlights the dominance of diesel and petrol fuel types."
            )

            
            st.write("### Distribution of the number of previous owners")
            owner_types = df_v1["owner"].unique()
            owner_counts = df_v1["owner"].value_counts()
            f, ax = plt.subplots(figsize=(5, 5))
            ax.pie(owner_counts, labels=owner_types, autopct="%1.1f%%", startangle=0)
            ax.set_title("Distribution of the number of previous owners")
            st.pyplot(f)
            st.write(
                "This pie chart illustrates the distribution of cars based on the number of previous owners." 
                "The chart visually emphasizes that a significant majority of the cars are from their first or second owners.")

        with col2:
            st.write("### Seller Type Distribution")
            types = df_v1["seller_type"].unique()
            x = df_v1["seller_type"].value_counts()
            f, ax = plt.subplots(figsize=(5, 5))
            ax.pie(x, labels=types, autopct="%1.1f%%", startangle=90)
            ax.set_title("Seller Type Distribution")
            st.pyplot(f)
            st.write(
                "The pie chart displays the distribution of seller types for used cars."
                "It highlights that the majority of cars are sold by individual sellers, accounting for 83.2% of the market."
                "Dealers represent 13.8% of the sales, while Trustmark Dealers account for a smaller portion at 2.9%." 
                "This indicates that individual sellers dominate the used car market."
            )

            st.write("### Record Count by Company")
            companies = df_v1["Company"].unique()
            count = df_v1["Company"].value_counts()
            f, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=companies, y=count.values, ax=ax)
            ax.set_xlabel("Company Name")
            ax.set_ylabel("Count")
            ax.set_xticklabels(companies, rotation=75)
            st.pyplot(f)
            st.write(
                "The bar chart displays the quantity of purchased cars available from various brands."
                "Maruti leads with the highest number of cars, surpassing 2000 units."
                "Skoda, Hyundai, and Toyota follow, each with a significant quantity." 
                "The chart highlights a stark drop-off in quantity as the list progresses to less common brands like Jaguar, Volvo, and Peugeot."
            )

            st.write("### Transmission Type Distribution")
            types_transmission = df_v1["transmission"].unique()
            x_transmission = df_v1["transmission"].value_counts()
            f, ax = plt.subplots(figsize=(5, 5))
            ax.pie(x_transmission, labels=types_transmission, autopct="%1.1f%%", startangle=90)
            ax.set_title("Transmission Type Distribution")
            st.pyplot(f)

            st.write(
                "The pie chart shows the distribution of transmission types among the vehicles. Manual transmissions are overwhelmingly popular, comprising 87.1% of the cars." 
                "Automatic transmissions make up 12.9%. This suggests a strong preference for manual transmission in the dataset."
            )
                    
            st.write("### Distribution of car models (Top 30)")
            models = df_v1["Model"].unique()[:30]
            model_counts = df_v1["Model"].value_counts()[:30]
            f, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=models, y=model_counts.values, ax=ax)
            ax.set_xlabel("Car Model (Top 30)")
            ax.set_ylabel("Count")
            ax.set_xticklabels(models, rotation=75)
            ax.set_title("Most Common Car Models (Top 30)")
            st.pyplot(f)
            st.write(
                "This bar plot shows the top 30 car models based on the quantity available in the dataset."
                "The plot highlights the substantial popularity of the Swift model, with a sharp decline in numbers for other models. The x-axis labels are rotated for clarity."
                    )

        st.write("### Comparison of Price Distributions for Different Fuel Types")
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='fuel', y='selling_price', data=df_v1)
        plt.title("Selling Price Distribution for Different Fuel Types")
        plt.xlabel("Fuel Type")
        plt.ylabel("Selling Price")
        st.pyplot(plt)
        st.markdown(
            """
            **This box plot illustrates the selling price distribution across various fuel types: Diesel, Petrol, LPG, and CNG. The chart reveals:**
            - Diesel cars exhibit a wide range of prices, with most prices concentrated around 0.2 million.
            - Petrol vehicles also show a moderate price variation but with fewer outliers compared to diesel.
            - LPG and CNG cars have more compressed price ranges with significantly fewer outlying values.
            """
        )

        # 2. Comparison of Price Distributions for Different Transmission Types
        st.write("### Comparison of Price Distributions for Different Transmission Types")
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='transmission', y='selling_price', data=df_v1)
        plt.title("Selling Price Distribution for Different Transmission Types")
        plt.xlabel("Transmission Type")
        plt.ylabel("Selling Price")
        st.pyplot(plt)
        st.markdown(
            """
            **This box plot displays the selling price distribution for cars with Manual and Automatic transmissions. Key observations include:**
            - Automatic transmission cars are generally sold at significantly higher prices compared to their manual counterparts.
            - There is a greater distribution of outliers in the automatic segment, suggesting some very high-priced automatic cars.
            - Manual transmission cars have a more condensed price range, indicating a more consistent pricing trend.
            """
        )
        

    with st.expander("Hypothesis", expanded=True):
        st.write("### Hypothesis: Petrol cars with automatic transmission, sold by dealers, have a higher price than those sold by individuals")

        group_dealer = df_v1[(df_v1['fuel'] == 'Petrol') & (df_v1['transmission'] == 'Automatic') & (df_v1['seller_type'] == 'Dealer')][
            'selling_price']
        group_individual = df_v1[(df_v1['fuel'] == 'Petrol') & (df_v1['transmission'] == 'Automatic') & (df_v1['seller_type'] == 'Individual')][
            'selling_price']

        # 2. Comparison of Mean Prices
        st.write("#### Comparison of Average Selling Prices for Two Groups: Dealer vs Individual")
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='seller_type', y='selling_price',
                    data=df_v1[(df_v1['fuel'] == 'Petrol') & (df_v1['transmission'] == 'Automatic')])
        plt.title("Selling Price Distribution for Petrol Cars with Automatic Transmission")
        plt.xlabel("Seller Type")
        plt.ylabel("Selling Price")
        st.pyplot(plt)

        t_stat, p_value = stats.ttest_ind(group_dealer, group_individual, equal_var=False)

        st.write(f"t-test: t-statistic = {t_stat:.2f}, p-value = {p_value:.4f}")

        # 4. Interpretation of Results
        if p_value < 0.05:
            st.success(
                "The hypothesis is confirmed: petrol cars with automatic transmission, sold by dealers, have a higher price than those sold by individuals.")
        else:
            st.warning("The hypothesis is refuted: there is no statistically significant difference in selling prices between the groups.")
        st.markdown(
            """
            **The box plot displays the distribution of selling prices for petrol cars with automatic transmissions categorized by seller type: Individual, Dealer, and Trustmark Dealer.**
            - Individual Sellers: The pricing spread is wide, with most prices clustered at the lower end, indicated by several outliers extending above the interquartile range.
            - Dealers: Show a higher median price compared to individuals, with a more concentrated interquartile range suggesting less variability in pricing.
            - Trustmark Dealers: Also have higher median prices than individual sellers, but with fewer extreme outliers compared to the Dealer category.
            
            **The statistical test results confirm the hypothesis that petrol cars with automatic transmission sold by dealers have higher prices than those sold by individuals,** 
            **indicating a significant price difference across seller types.**            
            """
        )

    st.write("### Dataset for Model Training")
    st.markdown(
        """
        **How was the dataset processed?**
        - Categorical attributes like “owner”, “fuel”, “seller_type”, “transmission”, “name” were replaced with numeric values. For example the value of “Diesel” was replaced with “1.0” and “Petrol” with “2.0”.
        - The “mileage” attribute contains values in km/kg and kmpl. For preprocessing km/kg were converted to kmpl.
        - For the engine and max_power attributes, the lowercase values were removed from the end.
        - The attribute “torque” was removed because it was difficult to process it.           
        """
    )

    st.write("[owner, fuel, seller_type, transmission, name] -> to digits\n")
    df_v2 = pd.read_csv('data/data_preproc_v2.csv')
    st.write(df_v2.head(5))


def show_prediction_page():
    st.header("Enter Car Parameters")
    user_input_df = main_page_input_features()

    if st.button("Make Prediction"):
        prediction = model_prediction(user_input_df)
        write_user_data(user_input_df)
        write_prediction(prediction)


def main_page_input_features():
    col1, col2 = st.columns(2)

    with col1:
        name = ""
        year = st.slider("Year of manufacture", min_value=1950, max_value=2023, value=2015, step=1)
        km_driven = st.slider("km driven", min_value=0, max_value=1000000, value=50000, step=1)
        fuel = st.selectbox("Fuel type", ("Diesel", "Petrol", "LPG", "CNG"))
        seller_type = st.selectbox("Seller type", ("Individual", "Dealer", "Trustmark Dealer"))
        seats = st.slider("Seats", min_value=2, max_value=14, value=5, step=1)

    with col2:
        transmission = st.selectbox("Transmission", ("Automatic", "Manual"))
        owner = st.selectbox("Owner", ("First", "Second", "Third", "Fourth & Above", "Test Drive car"))
        mileage = st.slider("Mileage, kmpl", min_value=0, max_value=50, value=18, step=1)
        engine = st.slider("Engine, CC", min_value=500, max_value=3600, value=1000, step=1)
        max_power = st.slider("Max power, bhp", min_value=0, max_value=14, value=7, step=1)

    translatetion = {
        "Diesel": 1,
        "Petrol": 2,
        "LPG": 3,
        "CNG": 4,
        "Individual": 1,
        "Dealer": 2,
        "Trustmark Dealer": 3,
        "Manual": 1,
        "Automatic": 2,
        "First": 1,
        "Second": 2,
        "Third": 3,
        "Fourth & Above": 4,
        "Test Drive car": 5,
    }

    data = {
        "name": prep_name(name),
        "year": year,
        "km_driven": km_driven,
        "fuel": translatetion[fuel],
        "seller_type": translatetion[seller_type],
        "transmission": translatetion[transmission],
        "owner": translatetion[owner],
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
    }

    df = pd.DataFrame(data, index=[0])
    return df


def write_user_data(df):
    st.write("## Your Data")
    st.write(df)


def write_prediction(prediction):
    st.write("## Prediction")
    st.write(prediction)


def prep_name(name):
    with open('models.json', 'r') as json_file:
        d = json.load(json_file)
    for i in d.keys():
        if name.lower().strip() in i.lower():
            return d[i]
    else:
        return 0


if __name__ == "__main__":
    process_main_page()
