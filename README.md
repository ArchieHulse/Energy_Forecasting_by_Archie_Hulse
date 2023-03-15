   # Energy Forecasting 
   by Archie Hulse

![WindFloat-Atlantic_Principle-Power](https://user-images.githubusercontent.com/122451494/225095974-8b68699d-b64a-4357-a24c-055f4eb79cd6.jpeg)
<sup>*(WindFloat Atlantic offshore wind farm, situated 12km off the coast of Portugal)*</sup>

---
# [LINK TO PRESENTATION](https://arcg.is/0WvD8n)
---
# *Study Objective:*

The main aim of this study was to predict the electricity production of 3 offshore wind turbines up to 5 days ahead of time.
---
This would be done using performance data from a real offshore wind turbine called T11 which is part of the WindFloat Atlantic array situated 12km off the coast of Portugal.

The turbine performance (SCADA) data consited of the Total Active Power (kWh) of the turbine recorded every 12 minutes from 2017-18, with over 39 thousand entries. Data was sourced from EDP energy (https://www.edp.com/en/edp-stories/offshore-wind).

Then cleaning and fitting of said data into a neural network was carried out using SQL and Python code.
The model was also given data of the corresponding wind speed and wind direction at 100m altitude for the same location as the turbine and at the same times (41.686371 N, -8.996471 W), (year 2017-18).

I chose the WindFloat Atlantic Turbine to base my model predictions on since it is the world's first and largest semi-submersible floating offshore wind farm and produces total power capacity of 25(MW), equivalent to the energy consumed by 60 thousand families over 1 year.

---
# *DATA:*

The turbine performance (SCADA) data consited of the Total Active Power (kWh) of the turbine recorded every 12 minutes from 2017-18, with over 50 thousand entries. Data was sourced from EDP energy (https://www.edp.com/en/edp-stories/offshore-wind).

Meteorological wind speed data, at 100m altitude in (m/s), for the entire European basin polygon at a resolution of 0.3x0.3km squared was obtained through API calls from Meteomatics (https://www.meteomatics.com/en/api/url-creator/), with more than 160 thousand entries.

Meteorological wind speed and direction data, at 100m altitude in (m/s) and (° absolute), for the 3 specific locations of:
                        - WindFloat Atlantic Turbine T11 (41.686371 N, -8.996471 W), Portugal (the one used to train the model)
                        - Spanish Government Proposed Menorca Wind farm (40.97 N, 3.98 W), Mediterranean Sea
                        - Spanish Government Proposed Galicia Wind farm (44.60 N, -7.80 W), North Atlantic
                        
                        
<img width="862" alt="Screenshot 2023-03-15 at 10 07 03" src="https://user-images.githubusercontent.com/122451494/225266767-90d41269-7863-4d71-8ca2-b7488254255d.png">

<img width="423" alt="Screenshot 2023-03-15 at 10 09 05" src="https://user-images.githubusercontent.com/122451494/225267046-fc48ce58-bbd0-45d2-9efe-6e9df831c143.png">

![MAP1](https://user-images.githubusercontent.com/122451494/225286659-fe9c507a-4d57-4030-b751-c4f9ca020661.gif)
                        
---
# *Neural Network*

I developed a neural network model, using Python Code, called a Multilayer Perceptron (MLP) Regressor for my energy forecast predictions. I chose this model since it adapts to predicting time series data very well and calculated an over all Mean Squared Error (MAE) of 10% from the test data.

The model is a type of feedforward neural network, where the information flows in one direction, from the input layer through one or more hidden layers to the output layer. The MLP Regressor can handle complex non-linear relationships between the input variables and the target variable.

            TP = pd.read_csv("../DATA/turbine_Performance_prediction_dataframe.csv",)

            #Convert the datetime column to the index of the DataFrame:
            TP['Time'] = pd.to_datetime(TP['Time'])
            TP.set_index('Time', inplace=True)

            #Drop columns not needed for model prediction:
            TP_MODEL = TP.drop(columns=['Unnamed: 0'])

            # Split the data into training and testing sets:
            X_train, X_test, y_train, y_test = train_test_split(TP_MODEL[['WindSpeed (m/s)',
            'Wind Direction (°)']], TP_MODEL['Total Active Power (kWh)'],
            test_size=0.2, random_state=42)

            # Create an instance of the MLPRegressor class:
            model = MLPRegressor(hidden_layer_sizes=(100,50), activation='relu',
            solver='adam', learning_rate='adaptive', max_iter=500)

            # Fit the model to the training data:
            model.fit(X_train, y_train)

            # Use the trained model to make predictions on the testing data:
            y_pred = model.predict(X_test)

            # Save the trained model to a file
            joblib.dump(model, 'trained_model.pkl')

            # Calculate the RMSE (root mean squared error) and R2 (R-squared) values
            # between the predicted and actual values:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            print(f'RMSE: {rmse:.2f}, R2: {r2:.2f}, MAE: {mae:.2f}')

            # Use the model to predict the new data
            X_new = forecast_data[['WindSpeed (m/s)', 'Wind Direction (°)']]
            y_new = list(model.predict(X_new))


            RMSE: 14.76, R2: 0.98, MAE: 10.53


- Above is my Python code used to train the MLP Regressor. I achieved this by splitting my data into 80% actual and 20% test for the model to predict.
- The Root Mean Squared Error (RMSE) and Root Squared (R2) values were calculated to show the level of accuracy in the models prediction power.
- RMSE score of 14.76, R2 score of 0.98 and a MAE score of 10.53
- Forecasted wind speed and direction data at 6 hour intervals, for the next 5 days was fitted to my model so that it could predict the corresponding power made by the turbine.

<img width="622" alt="Screenshot 2023-03-12 at 19 58 52" src="https://user-images.githubusercontent.com/122451494/225266461-e8fe8c05-cd16-441c-90dd-46da734ab5d8.png">

The predictions obtained from the model were then unscaled and plotted against the testing dataset to observe the deviation of our predicted values from actual values. Based on the visualisation, we can conclude that our model was able to successfully replicate the trend of the actual test dataset



















