#import inspect
from ensurepip import version
import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np# type: ignore
import base64 
from unicodedata import decimal# type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.svm import SVC# type: ignore
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix# type: ignore
from sklearn.ensemble import RandomForestClassifier# type: ignore
from sklearn.metrics import mean_squared_error# type: ignore
from sklearn.neighbors import KNeighborsClassifier# type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.neighbors import KNeighborsRegressor # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from PIL import Image 


SVMMax = 0
KNNMax = 0
RFMax = 0

st.set_page_config(page_title ="Crime Prediction", 
                       page_icon=':knife:', 
                       layout='centered')

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    The bg will be static and won't take resolution of device into account.
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
def get_base64_of_bin_file(bin_file):
    """
    function to read png file 
    ----------
    bin_file: png -> the background image in local folder
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    """
    function to display png as bg
    ----------
    png_file: png -> the background image in local folder
    """
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    st.App {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_bg_hack('backgroundcopy1.png')

st.title("Welcome to CSC649 Group Project")
st.write("This system will demonstrate a crime prediction investigation based on five different datasets which is Crime in Berlin, Germany dataset, Crime in Boston dataset, NYPD Complaint Data Current, Crime Against Women and Marijuan Crime Related.")
st.write("This system is using three different algorithms to train the model which are Support Vector Machine, K-Nearest Neighbors and Random Forest.")

logo = Image.open('background.png')
st.sidebar.image(logo, use_column_width=True) 
selectDataset = st.sidebar.selectbox ("Select Dataset", options = ["Select Dataset", "Crime in Berlin, Germany", "Crimes in Boston", "NYPD Complaint Data Current","Crime Against Women","Marijuana Crime Related","Conclusion"])

if selectDataset == "Crime in Berlin, Germany":

	st.subheader("Crime in Berlin, Germany: Crime type and Comparison of Machine Learning Algorithms for Classifying Crime Hotspots in the District of Berlin, Germany")
	st.subheader("Full Dataset For crime in Berlin, Germany")
	data1 = pd.read_csv ('Berlin_Crimes_Dataset_.csv')
	data1

	st.subheader("Data Input for crime in Berlin, Germany")
	data_input = data1.drop(columns = ['Unnamed: 0', 'Year','DistrictNew', 	'Code', 'LocationNew', 'Local'])
	data_input

	st.subheader("Data Target for crime in Berlin, Germany")
	data_target = data1['DistrictNew']
	data_target

	st.subheader ("Training and Testing Data will be divided using Train_Test_Split library")
	input_train,input_test,target_train, target_test = train_test_split	(data_input, data_target, test_size=0.2)
	
	st.subheader("Training data for input train and test")
	st.write("Training Data Input")
	input_train
	st.write("Testing data input")
	input_test


	selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model","Support Vector Machine", "K-Nearest Neighbors", "Random Forest","All Models"])

	if selectModel == "Support Vector Machine":

		st.subheader("Suppport Vector Machine for Crime Classification in Berlin, Germany")
		
		st.subheader("Support Vector Machine Crime Investigation Model")

		whole = st.checkbox('Run for whole dataset')
		inpuser = st.checkbox('Run using input dataset')

		svm1 = SVC (kernel = 'linear')
		svm2 = SVC (kernel = 'rbf')
		svm3 = SVC (kernel = 'poly')
		svm4 = SVC (kernel = 'sigmoid')

		st.write("Training the Model...")
		svm1.fit (input_train, target_train)
		svm2.fit (input_train, target_train)
		svm3.fit (input_train, target_train)
		svm4.fit (input_train, target_train)

		if whole:

			st.write("Successfully Train the Model")
		
			outputPredictedlinear = svm1.predict(input_test)
			outputPredictedrbf = svm2.predict(input_test)
			outputPredictedpoly = svm3.predict(input_test)
			outputPredictedsig = svm4.predict(input_test)
			st.write("Predicted Results for Testing Dataset - Linear :")
			outputPredictedlinear
			st.write("Predicted Results for Testing Dataset - RBF :")
			outputPredictedrbf
			st.write("Predicted Results for Testing Dataset - Polynomial :")
			outputPredictedpoly
			st.write("Predicted Results for Testing Dataset - Sigmoid :")
			outputPredictedsig

			acclinear = accuracy_score (outputPredictedlinear, target_test)
			percentageacclinear = "{:.0%}".format(acclinear) 
			accrbf = accuracy_score (outputPredictedrbf, target_test)
			percentageaccrbf = "{:.0%}".format(accrbf) 
			accpoly = accuracy_score (outputPredictedpoly, target_test)
			percentageaccpoly = "{:.0%}".format(accpoly) 
			accsig = accuracy_score (outputPredictedsig, target_test)
			percentageaccsig = "{:.0%}".format(accsig) 
			st.write("The Accuracy Score Produced by Linear kernel: ", percentageacclinear)
			st.write("The Accuracy Score Produced by RBF kernel: ", percentageaccrbf)
			st.write("The Accuracy Score Produced by Polynomial kernel: ", percentageaccpoly)
			st.write("The Accuracy Score Produced by Sigmoid kernel: ", percentageaccsig)
		
		elif inpuser:

			robery = st.text_input("Number of robbery: ")
			streetrobery = int(st.text_input("Number of street robbery: "))
			injury = int(st.text_input("Number of injury: "))
			assault = int(st.text_input("Number of assault: "))
			threat = int(st.text_input("Number of threat: "))
			theft = int(st.text_input("Number of theft: "))
			ctheft = int(st.text_input("Number of car theft: "))
			actheft = int(st.text_input("Number of auto car theft: "))
			btheft = int(st.text_input("Number of bike theft: "))
			burg = int(st.text_input("Number of burglary: "))
			fire = int(st.text_input("Number of fire crime: "))
			arsony = int(st.text_input("Number of arson: "))
			damage = int(st.text_input("Number of damage crime: "))
			graf = int(st.text_input("Number of graffitu crime: "))
			drug = int(st.text_input("Number of drug crime: "))

			robery1 = int(robery)

			st.write("Successfully Train the Model")
			outputPredictedlinear = svm1.predict([[robery,streetrobery,injury,assault,threat,theft,theft,ctheft,actheft,btheft,burg,fire,arsony,damage,graf,drug]])
			outputPredictedrbf = svm2.predict([[robery,streetrobery,injury,assault,threat,theft,theft,ctheft,actheft,btheft,burg,fire,arsony,damage,graf,drug]])
			outputPredictedpoly = svm3.predict([[robery,streetrobery,injury,assault,threat,theft,theft,ctheft,actheft,btheft,burg,fire,arsony,damage,graf,drug]])
			outputPredictedsig = svm4.predict([[robery,streetrobery,injury,assault,threat,theft,theft,ctheft,actheft,btheft,burg,fire,arsony,damage,graf,drug]])
			
			acclinear = accuracy_score (outputPredictedlinear, target_test)
			percentageacclinear = "{:.0%}".format(acclinear) 
			accrbf = accuracy_score (outputPredictedrbf, target_test)
			percentageaccrbf = "{:.0%}".format(accrbf) 
			accpoly = accuracy_score (outputPredictedpoly, target_test)
			percentageaccpoly = "{:.0%}".format(accpoly) 
			accsig = accuracy_score (outputPredictedsig, target_test)
			percentageaccsig = "{:.0%}".format(accsig) 
			st.write("The Accuracy Score Produced by Linear kernel: ", percentageacclinear)
			st.write("The Accuracy Score Produced by RBF kernel: ", percentageaccrbf)
			st.write("The Accuracy Score Produced by Polynomial kernel: ", percentageaccpoly)
			st.write("The Accuracy Score Produced by Sigmoid kernel: ", percentageaccsig)
		

	if selectModel == "K-Nearest Neighbors":

		st.subheader("K-Nearest Neighbors for Crime Classification in Berlin, Germany")	

		

		knn = KNeighborsClassifier (n_neighbors = 1)
		knn1 = KNeighborsClassifier (n_neighbors = 10)
		knn2 = KNeighborsClassifier (n_neighbors = 20)
		knn3 = KNeighborsClassifier (n_neighbors = 30)

		st.write("Training the Model...")
		knn.fit (input_train, target_train)
		knn1.fit (input_train, target_train)
		knn2.fit (input_train, target_train)
		knn3.fit (input_train, target_train)

		whole = st.checkbox('Run for whole dataset')
		inpuser = st.checkbox('Run using input dataset')	

		if whole:

			st.write("Successfully Train the Model")
		
			outputPredictedKNN = knn.predict(input_test)
			st.write("Predicted Results for Testing Dataset - 1NN:")
			outputPredictedKNN
			outputPredictedKNN1 = knn1.predict(input_test)
			st.write("Predicted Results for Testing Dataset - 10NN:")
			outputPredictedKNN1
			outputPredictedKNN2 = knn2.predict(input_test)
			st.write("Predicted Results for Testing Dataset - 20NN:")
			outputPredictedKNN2
			outputPredictedKNN3 = knn3.predict(input_test)
			st.write("Predicted Results for Testing Dataset - 30NN:")
			outputPredictedKNN3

			accKNN = accuracy_score (outputPredictedKNN, target_test)
			percentageaccknn = "{:.0%}".format(accKNN) 
			st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 1: ", percentageaccknn)
			accKNN1 = accuracy_score (outputPredictedKNN1, target_test)
			percentageaccknn1 = "{:.0%}".format(accKNN1) 
			st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 10: ", percentageaccknn1)
			accKNN2 = accuracy_score (outputPredictedKNN2, target_test)
			percentageaccknn2 = "{:.0%}".format(accKNN2) 
			st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 20: ", percentageaccknn2)
			accKNN3 = accuracy_score (outputPredictedKNN3, target_test)
			percentageaccknn3 = "{:.0%}".format(accKNN3) 
			st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 30: ", percentageaccknn3)

		elif inpuser:

			knn = KNeighborsClassifier (n_neighbors = 1)
			knn1 = KNeighborsClassifier (n_neighbors = 10)
			knn2 = KNeighborsClassifier (n_neighbors = 20)
			knn3 = KNeighborsClassifier (n_neighbors = 30)

			st.write("Training the Model...")
			knn.fit (input_train, target_train)
			knn1.fit (input_train, target_train)
			knn2.fit (input_train, target_train)
			knn3.fit (input_train, target_train)

			robery = st.text_input("Number of robbery")
			streetrobery = st.text_input("Number of street robbery:")
			injury = st.text_input("Number of injury: ")
			assault = st.text_input("Number of assault: ")
			threat = st.text_input("Number of threat: ")
			theft = st.text_input("Number of theft: ")
			ctheft = st.text_input("Number of car theft: ")
			actheft = st.text_input("Number of auto car theft: ")
			btheft = st.text_input("Number of bike theft: ")
			burg = st.text_input("Number of burglary: ")
			fire = st.text_input("Number of fire crime: ")
			arsony = st.text_input("Number of arson: ")
			damage = st.text_input("Number of damage crime: ")
			graf = st.text_input("Number of graffitu crime: ")
			drug = st.text_input("Number of drug crime: ")

			rober = int(robery)
			rober1 = int(streetrobery)
			rober2 = int(injury)
			rober3 = int(assault)
			rober4 = int(threat)
			rober5 = int(theft)
			rober6 = int(ctheft)
			rober7 = int(actheft)
			rober8 = int(btheft)
			rober9 = int(burg)
			rober12 = int(fire)
			rober123 = int(arsony)
			rober15 = int(damage)
			rober13 = int(graf)
			rober14 = int(drug)

			uoutputPredictedKNN = knn.predict([[rober,rober1,rober2,rober3,rober4,rober5,rober6,rober7,rober8,rober9,rober12,rober123,rober15,rober13, rober14]])
			uoutputPredictedKNN1 = knn1.predict([[rober,rober1,rober2,rober3,rober4,rober5,rober6,rober7,rober8,rober9,rober12,rober123,rober15,rober13, rober14]])
			uoutputPredictedKNN2 = knn2.predict([[rober,rober1,rober2,rober3,rober4,rober5,rober6,rober7,rober8,rober9,rober12,rober123,rober15,rober13, rober14]])
			uoutputPredictedKNN3 = knn3.predict([[rober,rober1,rober2,rober3,rober4,rober5,rober6,rober7,rober8,rober9,rober12,rober123,rober15,rober13, rober14]])

			st.write("Successfully Train the Model")

			uaccKNN = accuracy_score (uoutputPredictedKNN, target_test)
			upercentageaccknn = "{:.0%}".format(uaccKNN) 
			st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 1: ", upercentageaccknn)
			uaccKNN1 = accuracy_score (uoutputPredictedKNN1, target_test)
			upercentageaccknn1 = "{:.0%}".format(uaccKNN1) 
			st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 10: ", upercentageaccknn1)
			uaccKNN2 = accuracy_score (uoutputPredictedKNN2, target_test)
			upercentageaccknn2 = "{:.0%}".format(uaccKNN2) 
			st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 20: ", upercentageaccknn2)
			uaccKNN3 = accuracy_score (uoutputPredictedKNN3, target_test)
			upercentageaccknn3 = "{:.0%}".format(uaccKNN3) 
			st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 30: ", upercentageaccknn3)
		
	if selectModel == "Random Forest":
		
		st.subheader("Random Forest for Crime Classification in Berlin, Germany")

		whole = st.checkbox('Run for whole dataset')
		inpuser = st.checkbox('Run using input dataset')

		rf = RandomForestClassifier (n_estimators = 50, random_state = 0)
		rf1 = RandomForestClassifier (n_estimators = 100, random_state = 0)
		rf2 = RandomForestClassifier (n_estimators = 200, random_state = 0)
		rf3 = RandomForestClassifier (n_estimators = 300, random_state = 0)

		st.write("Training the Model...")
		rf.fit (input_train, target_train)
		rf1.fit (input_train, target_train)
		rf2.fit (input_train, target_train)
		rf3.fit (input_train, target_train)

		if whole:
		
			st.write("Successfully Train the Model")

			outputPredicted50 = rf.predict(input_test)
			st.write("Predicted Results for Testing Dataset:")
			outputPredicted50
			outputPredicted100 = rf1.predict(input_test)
			st.write("Predicted Results for Testing Dataset:")
			outputPredicted100
			outputPredicted200 = rf2.predict(input_test)
			st.write("Predicted Results for Testing Dataset:")
			outputPredicted200
			outputPredicted300 = rf3.predict(input_test)
			st.write("Predicted Results for Testing Dataset:")
			outputPredicted300

			acc50 = accuracy_score (outputPredicted50, target_test)
			percentageacc50 = "{:.0%}".format(acc50) 
			st.write("The Accuracy Score Produced by n_estimator = 50: ", percentageacc50)
			acc100 = accuracy_score (outputPredicted100, target_test)
			percentageacc100 = "{:.0%}".format(acc100) 
			st.write("The Accuracy Score Produced by n_estimator = 100: ", percentageacc100)
			acc200 = accuracy_score (outputPredicted200, target_test)
			percentageacc200 = "{:.0%}".format(acc50) 
			st.write("The Accuracy Score Produced by n_estimator = 200: ", percentageacc200)
			acc300 = accuracy_score (outputPredicted300, target_test)
			percentageacc300 = "{:.0%}".format(acc50) 
			st.write("The Accuracy Score Produced by n_estimator = 300: ", percentageacc300)

		elif inpuser:

			robery = int(st.text_input("Number of robbery: "))
			streetrobery = int(st.text_input("Number of street robbery: "))
			injury = int(st.text_input("Number of injury: "))
			assault = int(st.text_input("Number of assault: "))
			threat = int(st.text_input("Number of threat: "))
			theft = int(st.text_input("Number of theft: "))
			ctheft = int(st.text_input("Number of car theft: "))
			actheft = int(st.text_input("Number of auto car theft: "))
			btheft = int(st.text_input("Number of bike theft: "))
			burg = int(st.text_input("Number of burglary: "))
			fire = int(st.text_input("Number of fire crime: "))
			arsony = int(st.text_input("Number of arson: "))
			damage = int(st.text_input("Number of damage crime: "))
			graf = int(st.text_input("Number of graffitu crime: "))
			drug = int(st.text_input("Number of drug crime: "))

			st.write("Successfully Train the Model")

			outputPredicted50 = rf.predict([[robery,streetrobery,injury,assault,threat,theft,theft,ctheft,actheft,btheft,burg,fire,arsony,damage,graf,drug]])
			outputPredicted100 = rf1.predict([[robery,streetrobery,injury,assault,threat,theft,theft,ctheft,actheft,btheft,burg,fire,arsony,damage,graf,drug]])
			outputPredicted200 = rf2.predict([[robery,streetrobery,injury,assault,threat,theft,theft,ctheft,actheft,btheft,burg,fire,arsony,damage,graf,drug]])
			outputPredicted300 = rf3.predict([[robery,streetrobery,injury,assault,threat,theft,theft,ctheft,actheft,btheft,burg,fire,arsony,damage,graf,drug]])

			acc50 = accuracy_score (outputPredicted50, target_test)
			percentageacc50 = "{:.0%}".format(acc50) 
			st.write("The Accuracy Score Produced by n_estimator = 50: ", percentageacc50)
			acc100 = accuracy_score (outputPredicted100, target_test)
			percentageacc100 = "{:.0%}".format(acc100) 
			st.write("The Accuracy Score Produced by n_estimator = 100: ", percentageacc100)
			acc200 = accuracy_score (outputPredicted200, target_test)
			percentageacc200 = "{:.0%}".format(acc200) 
			st.write("The Accuracy Score Produced by n_estimator = 200: ", percentageacc200)
			acc300 = accuracy_score (outputPredicted300, target_test)
			percentageacc300 = "{:.0%}".format(acc300) 
			st.write("The Accuracy Score Produced by n_estimator = 300: ", percentageacc300)	

elif selectDataset == "Crimes in Boston":

	st.subheader("Dataset2: Machine Learning Algorithms for Crime Analysis, Classification and Prediction of Crime in Boston")
	st.subheader("Full Dataset For Crime in Boston")
	data = pd.read_csv ('cromeboston.csv',encoding='cp1252')
	data

	def tran_dayofweek(x):
		if x == 'Monday':
			return 1
		if x == 'Tuesday':
			return 2
		if x == 'Wednesday':
			return 3
		if x == 'Thursday':
			return 4
		if x == 'Friday':
			return 5
		if x == 'Saturday':
			return 6
		if x == 'Sunday':
			return 7
	
	data['Trans_DayOfWeek'] = data['DAY_OF_WEEK'].apply(tran_dayofweek)

	st.subheader("Data Input for  Crime in Boston")
	data_input = data.drop(columns=['INCIDENT_NUMBER','OFFENSE_CODE_GROUP','OFFENSE_DESCRIPTION','DISTRICT','REPORTING_AREA','DAY_OF_WEEK','UCR_PART','STREET','Lat','Long'])  
	data_input

	st.subheader("Data Target for Cases under Crime in Boston")
	data_target = data['DISTRICT']
	data_target

	st.subheader ("Training and Testing Data will be divided using Train_Test_Split")
	input_train, input_test, target_train, target_test = train_test_split(data_input, data_target, test_size=0.2)

	st.subheader("Training data for input and target")
	st.write("Training Data Input")
	input_train
	st.write("Training Data Target")
	target_train

	st.subheader("Testing data for input and target")
	st.write("Testing Data Input")
	input_test
	st.write("Testing Data Target")
	target_test

	selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])


	if selectModel == "Support Vector Machine":

		st.subheader("Support Vector Machine Crime Investigation Model")

		
	
		svm1 = SVC (kernel = 'linear')
		svm2 = SVC (kernel = 'rbf')
		svm3 = SVC (kernel = 'poly')
		svm4 = SVC (kernel = 'sigmoid')

		st.write("Training the Model...")
		svm1.fit (input_train, target_train)
		svm2.fit (input_train,target_train)
		svm3.fit (input_train, target_train)
		svm4.fit (input_train, target_train)

		st.write("Successfully Train the Model")

		outputPredictedlinear = svm1.predict(input_test)
		outputPredictedrbf = svm2.predict(input_test)
		outputPredictedpoly = svm3.predict(input_test)
		outputPredictedsig = svm4.predict(input_test)
		st.write("Predicted Results for Testing Dataset - Linear :")
		outputPredictedlinear
		st.write("Predicted Results for Testing Dataset - RBF :")
		outputPredictedrbf
		st.write("Predicted Results for Testing Dataset - Polynomial :")
		outputPredictedpoly
		st.write("Predicted Results for Testing Dataset - Sigmoid :")
		outputPredictedsig

		acclinear = accuracy_score (outputPredictedlinear, target_test)
		accrbf = accuracy_score (outputPredictedrbf, target_test)
		accpoly = accuracy_score (outputPredictedpoly, target_test)
		accsig = accuracy_score (outputPredictedsig, target_test)
		st.write("The Accuracy Score Produced by Linear kernel: ", acclinear.round(decimals = 2))
		st.write("The Accuracy Score Produced by RBF kernel: ", accrbf.round(decimals = 2))
		st.write("The Accuracy Score Produced by Polynomial kernel: ", accpoly.round(decimals = 2))
		st.write("The Accuracy Score Produced by Sigmoid kernel: ", accsig.round(decimals = 2))

	elif selectModel == "Random Forest":

		st.subheader("Random Forest Crime Investigation Model")
	
		rf = RandomForestClassifier (n_estimators = 50, random_state = 0)
		rf1 = RandomForestClassifier (n_estimators = 100, random_state = 0)
		rf2 = RandomForestClassifier (n_estimators = 200, random_state = 0)
		rf3 = RandomForestClassifier (n_estimators = 300, random_state = 0)

		st.write("Training the Model...")
		rf.fit (input_train, target_train)
		rf1.fit (input_train, target_train)
		rf2.fit (input_train, target_train)
		rf3.fit (input_train, target_train)

		st.write("Successfully Train the Model")
		
		outputPredicted50 = rf.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted50
		outputPredicted100 = rf1.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted100
		outputPredicted200 = rf2.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted200
		outputPredicted300 = rf3.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted300

		acc50 = accuracy_score (outputPredicted50, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 50: ", acc50.round(decimals = 2))
		acc100 = accuracy_score (outputPredicted100, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 100: ", acc100.round(decimals = 2))
		acc200 = accuracy_score (outputPredicted200, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 200: ", acc200.round(decimals = 2))
		acc300 = accuracy_score (outputPredicted300, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 300: ", acc300.round(decimals = 2))

	elif selectModel == "K-Nearest Neighbors":

		st.subheader("K-Nearest Neighbors Crime Investigation Model")
	
		knn = KNeighborsClassifier (n_neighbors = 1)
		knn1 = KNeighborsClassifier (n_neighbors = 10)
		knn2 = KNeighborsClassifier (n_neighbors = 20)
		knn3 = KNeighborsClassifier (n_neighbors = 30)

		st.write("Training the Model...")
		knn.fit (input_train, target_train)
		knn1.fit (input_train, target_train)
		knn2.fit (input_train, target_train)
		knn3.fit (input_train, target_train)

		st.write("Successfully Train the Model")
		
		outputPredictedKNN = knn.predict(input_test)
		st.write("Predicted Results for Testing Dataset - 1NN:")
		outputPredictedKNN
		outputPredictedKNN1 = knn1.predict(input_test)
		st.write("Predicted Results for Testing Dataset - 10NN:")
		outputPredictedKNN1
		outputPredictedKNN2 = knn2.predict(input_test)
		st.write("Predicted Results for Testing Dataset - 20NN:")
		outputPredictedKNN2
		outputPredictedKNN3 = knn3.predict(input_test)
		st.write("Predicted Results for Testing Dataset - 3NN:")
		outputPredictedKNN3

		accKNN = accuracy_score (outputPredictedKNN, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 1: ", accKNN.round(decimals = 2))
		accKNN1 = accuracy_score (outputPredictedKNN1, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 10: ", accKNN1.round(decimals = 2))
		accKNN2 = accuracy_score (outputPredictedKNN2, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 20: ", accKNN2.round(decimals = 2))
		accKNN3 = accuracy_score (outputPredictedKNN3, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 30: ", accKNN3.round(decimals = 2))

elif selectDataset == "NYPD Complaint Data Current":

	st.subheader("Dataset 3: Crime Investigation Model Using Several Machine Learning Algorithm Based on Suspect and Victims Demographic")

	st.subheader("Full Dataset For NYPD Complaint Data")
	data_NYPD = pd.read_csv ('nypd-complaint-data-current-year-to-date1.csv')
	data_NYPD

	def tran_suspsex(x):
		if x == 'M':
			return 1
		if x == 'F':
			return 0

	data_NYPD['Trans_SuspSex'] = data_NYPD['SUSP_SEX'].apply(tran_suspsex)

	def tran_vicsex(x):
		if x == 'M':
			return 1
		if x == 'F':
			return 0

	data_NYPD['Trans_VicSex'] = data_NYPD['VIC_SEX'].apply(tran_vicsex)

	def tran_vicagegroup(x):
		if x == '<18':
			return 0
		if x == '18-24':
			return 1
		if x == '25-44':
			return 2
		if x == '45-64':
			return 3
		if x == '65+':
			return 4

	data_NYPD['Trans_VicAgeGroup'] = data_NYPD['VIC_AGE_GROUP'].apply(tran_vicagegroup)

	def tran_vicrace(x):
		if x == 'BLACK':
			return 0
		if x == 'WHITE':
			return 1
		if x == 'ASIAN/PAC.ISL':
			return 2
		if x == 'WHITE HISPANIC':
			return 3
		if x == 'BLACK HISPANIC':
			return 4
		if x == 'AMER IND':
			return 5

	data_NYPD['Trans_VicRace'] = data_NYPD['VIC_RACE'].apply(tran_vicrace)

	def tran_susprace(x):
		if x == 'BLACK':
			return 0
		if x == 'WHITE':
			return 1
		if x == 'ASIAN/PAC.ISL':
			return 2
		if x == 'WHITE HISPANIC':
			return 3
		if x == 'BLACK HISPANIC':
			return 4
		if x == 'AMER IND':
			return 5

	data_NYPD['Trans_SuspRace'] = data_NYPD['SUSP_RACE'].apply(tran_susprace)

	def tran_suspagegroup(x):
		if x == '<18':
			return 0
		if x == '18-24':
			return 1
		if x == '25-44':
			return 2
		if x == '45-64':
			return 3
		if x == '65+':
			return 4
    
	data_NYPD['Trans_SuspAgeGroup'] = data_NYPD['SUSP_AGE_GROUP'].apply(tran_suspagegroup)
    
	st.subheader("Data Input for NYPD")
	data_input = data_NYPD.drop(columns = ['CMPLNT_NUM','ADDR_PCT_CD','BORO_NM','CMPLNT_FR_DT','CMPLNT_FR_TM','CMPLNT_TO_DT','CMPLNT_TO_TM',
                               'CRM_ATPT_CPTD_CD','HADEVELOPT','HOUSING_PSA','JURISDICTION_CODE','JURIS_DESC','KY_CD','LAW_CAT_CD',
                               'LOC_OF_OCCUR_DESC','OFNS_DESC','PARKS_NM','PATROL_BORO','PD_CD','PD_DESC','PREM_TYP_DESC','RPT_DT',
                               'STATION_NAME','SUSP_AGE_GROUP','SUSP_RACE','SUSP_SEX','TRANSIT_DISTRICT','VIC_AGE_GROUP','VIC_RACE',
                               'VIC_SEX','X_COORD_CD','Y_COORD_CD','Latitude','Longitude','Lat_Lon'])
	data_input

	st.subheader("Data Target for NYPD")
	data_target = data_NYPD["OFNS_DESC"]
	data_target

	st.subheader ("Training and Testing Data will be divided using Train_Test_Split")
	X_train, X_test, y_train, y_test = train_test_split(data_input, data_target, test_size=0.2)

	st.subheader("Training data for input and target")
	st.write("Training Data Input")
	X_train
	st.write("Training Data Target")
	y_train

	st.subheader("Testing data for input and target")
	st.write("Testing Data Input")
	X_test
	st.write("Testing Data Target")
	y_test

	selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest","All Models"])


	if selectModel == "Support Vector Machine":

		st.subheader("Support Vector Machine Crime Investigation Model")
	
		svm1 = SVC (kernel = 'linear')
		svm2 = SVC (kernel = 'rbf')
		svm3 = SVC (kernel = 'poly')
		svm4 = SVC (kernel = 'sigmoid')

		st.write("Training the Model...")
		svm1.fit (X_train, y_train)
		svm2.fit (X_train, y_train)
		svm3.fit (X_train, y_train)
		svm4.fit (X_train, y_train)

		st.write("Successfully Train the Model")
		
		outputPredictedlinear = svm1.predict(X_test)
		outputPredictedrbf = svm2.predict(X_test)
		outputPredictedpoly = svm3.predict(X_test)
		outputPredictedsig = svm4.predict(X_test)
		st.write("Predicted Results for Testing Dataset - Linear :")
		outputPredictedlinear
		st.write("Predicted Results for Testing Dataset - RBF :")
		outputPredictedrbf
		st.write("Predicted Results for Testing Dataset - Polynomial :")
		outputPredictedpoly
		st.write("Predicted Results for Testing Dataset - Sigmoid :")
		outputPredictedsig

		acclinear = accuracy_score (outputPredictedlinear, y_test)
		accrbf = accuracy_score (outputPredictedrbf, y_test)
		accpoly = accuracy_score (outputPredictedpoly, y_test)
		accsig = accuracy_score (outputPredictedsig, y_test)
		st.write("The Accuracy Score Produced by Linear kernel: ", acclinear.round(decimals = 2))
		st.write("The Accuracy Score Produced by RBF kernel: ", accrbf.round(decimals = 2))
		st.write("The Accuracy Score Produced by Polynomial kernel: ", accpoly.round(decimals = 2))
		st.write("The Accuracy Score Produced by Sigmoid kernel: ", accsig.round(decimals = 2))

		#SVMMax = max(acclinear,accsig,accrbf,accpoly)

	elif selectModel == "Random Forest":

		st.subheader("Random Forest Crime Investigation Model")
	
		rf = RandomForestClassifier (n_estimators = 50, random_state = 0)
		rf1 = RandomForestClassifier (n_estimators = 100, random_state = 0)
		rf2 = RandomForestClassifier (n_estimators = 200, random_state = 0)
		rf3 = RandomForestClassifier (n_estimators = 300, random_state = 0)

		st.write("Training the Model...")
		rf.fit (X_train, y_train)
		rf1.fit (X_train, y_train)
		rf2.fit (X_train, y_train)
		rf3.fit (X_train, y_train)

		st.write("Successfully Train the Model")
		
		outputPredicted50 = rf.predict(X_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted50
		outputPredicted100 = rf1.predict(X_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted100
		outputPredicted200 = rf2.predict(X_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted200
		outputPredicted300 = rf3.predict(X_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted300

		acc50 = accuracy_score (outputPredicted50, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 50: ", acc50.round(decimals = 2))
		acc100 = accuracy_score (outputPredicted100, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 100: ", acc100.round(decimals = 2))
		acc200 = accuracy_score (outputPredicted200, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 200: ", acc200.round(decimals = 2))
		acc300 = accuracy_score (outputPredicted300, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 300: ", acc300.round(decimals = 2))

		RFMax = max(acc300,acc200,acc100,acc50)

	elif selectModel == "K-Nearest Neighbors":

		st.subheader("K-Nearest Neighbors Crime Investigation Model")
	
		knn = KNeighborsClassifier (n_neighbors = 1)
		knn1 = KNeighborsClassifier (n_neighbors = 10)
		knn2 = KNeighborsClassifier (n_neighbors = 20)
		knn3 = KNeighborsClassifier (n_neighbors = 30)

		st.write("Training the Model...")
		knn.fit (X_train, y_train)
		knn1.fit (X_train, y_train)
		knn2.fit (X_train, y_train)
		knn3.fit (X_train, y_train)

		st.write("Successfully Train the Model")
		
		outputPredictedKNN = knn.predict(X_test)
		st.write("Predicted Results for Testing Dataset - 1NN:")
		outputPredictedKNN
		outputPredictedKNN1 = knn1.predict(X_test)
		st.write("Predicted Results for Testing Dataset - 10NN:")
		outputPredictedKNN1
		outputPredictedKNN2 = knn2.predict(X_test)
		st.write("Predicted Results for Testing Dataset - 20NN:")
		outputPredictedKNN2
		outputPredictedKNN3 = knn3.predict(X_test)
		st.write("Predicted Results for Testing Dataset - 3NN:")
		outputPredictedKNN3

		accKNN = accuracy_score (outputPredictedKNN, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 1: ", accKNN.round(decimals = 2))
		accKNN1 = accuracy_score (outputPredictedKNN1, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 10: ", accKNN1.round(decimals = 2))
		accKNN2 = accuracy_score (outputPredictedKNN2, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 20: ", accKNN2.round(decimals = 2))
		accKNN3 = accuracy_score (outputPredictedKNN3, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 30: ", accKNN3.round(decimals = 2))

		KNNMax =  max( accKNN3, accKNN2, accKNN1, accKNN3)
		
	
	elif selectModel == "All Models":
		st.write("MAX",KNNMax, SVMMax, RFMax) 

elif selectDataset == "Crime Against Women":

	st.subheader("Dataset 4: The Comparison Three Algorithm in Predicting Crime Against Woman in India")
	st.subheader("Full Dataset For Crime Against Woman in India")
	data_crimeindia = pd.read_csv ('crime_under_women_streamlit.csv')
	data_crimeindia
    
	st.subheader("Data Input for Cases under Crime in India")
	data_inputindia = data_crimeindia.drop(columns = ['Area_Name','Year','Group_Name','Sub_Group_Name'])
	data_inputindia

	st.subheader("Data Target for Cases under Crime in India")
	data_targetindia = data_crimeindia['Year']
	data_targetindia

	st.subheader ("Training and Testing Data will be divided using Train_Test_Split")
	X_train, X_test, y_train, y_test = train_test_split(data_inputindia, data_targetindia, test_size=0.2)

	st.subheader("Training data for input and target")
	st.write("Training Data Input")
	X_train
	st.write("Training Data Target")
	y_train

	st.subheader("Testing data for input and target")
	st.write("Testing Data Input")
	X_test
	st.write("Testing Data Target")
	y_test

	selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])


	if selectModel == "Support Vector Machine":

		st.subheader("Support Vector Machine Crime Investigation Model")
	
		svm1 = SVC (kernel = 'linear')
		svm2 = SVC (kernel = 'rbf')
		svm3 = SVC (kernel = 'poly')
		svm4 = SVC (kernel = 'sigmoid')

		st.write("Training the Model...")
		svm1.fit (X_train, y_train)
		svm2.fit (X_train, y_train)
		svm3.fit (X_train, y_train)
		svm4.fit (X_train, y_train)

		st.write("Successfully Train the Model")
		
		outputPredictedlinear = svm1.predict(X_test)
		outputPredictedrbf = svm2.predict(X_test)
		outputPredictedpoly = svm3.predict(X_test)
		outputPredictedsig = svm4.predict(X_test)
		st.write("Predicted Results for Testing Dataset - Linear :")
		outputPredictedlinear
		st.write("Predicted Results for Testing Dataset - RBF :")
		outputPredictedrbf
		st.write("Predicted Results for Testing Dataset - Polynomial :")
		outputPredictedpoly
		st.write("Predicted Results for Testing Dataset - Sigmoid :")
		outputPredictedsig

		acclinear = accuracy_score (outputPredictedlinear, y_test)
		accrbf = accuracy_score (outputPredictedrbf, y_test)
		accpoly = accuracy_score (outputPredictedpoly, y_test)
		accsig = accuracy_score (outputPredictedsig, y_test)
		st.write("The Accuracy Score Produced by Linear kernel: ", acclinear.round(decimals = 2))
		st.write("The Accuracy Score Produced by RBF kernel: ", accrbf.round(decimals = 2))
		st.write("The Accuracy Score Produced by Polynomial kernel: ", accpoly.round(decimals = 2))
		st.write("The Accuracy Score Produced by Sigmoid kernel: ", accsig.round(decimals = 2))

	elif selectModel == "Random Forest":

		st.subheader("Random Forest Crime Investigation Model")
	
		rf = RandomForestClassifier (n_estimators = 50, random_state = 0)
		rf1 = RandomForestClassifier (n_estimators = 100, random_state = 0)
		rf2 = RandomForestClassifier (n_estimators = 200, random_state = 0)
		rf3 = RandomForestClassifier (n_estimators = 300, random_state = 0)

		st.write("Training the Model...")
		rf.fit (X_train, y_train)
		rf1.fit (X_train, y_train)
		rf2.fit (X_train, y_train)
		rf3.fit (X_train, y_train)

		st.write("Successfully Train the Model")
		
		outputPredicted50 = rf.predict(X_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted50
		outputPredicted100 = rf1.predict(X_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted100
		outputPredicted200 = rf2.predict(X_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted200
		outputPredicted300 = rf3.predict(X_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted300

		acc50 = accuracy_score (outputPredicted50, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 50: ", acc50.round(decimals = 2))
		acc100 = accuracy_score (outputPredicted100, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 100: ", acc100.round(decimals = 2))
		acc200 = accuracy_score (outputPredicted200, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 200: ", acc200.round(decimals = 2))
		acc300 = accuracy_score (outputPredicted300, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 300: ", acc300.round(decimals = 2))

	elif selectModel == "K-Nearest Neighbors":

		st.subheader("K-Nearest Neighbors Crime Investigation Model")
	
		knn = KNeighborsClassifier (n_neighbors = 1)
		knn1 = KNeighborsClassifier (n_neighbors = 10)
		knn2 = KNeighborsClassifier (n_neighbors = 20)
		knn3 = KNeighborsClassifier (n_neighbors = 30)

		st.write("Training the Model...")
		knn.fit (X_train, y_train)
		knn1.fit (X_train, y_train)
		knn2.fit (X_train, y_train)
		knn3.fit (X_train, y_train)

		st.write("Successfully Train the Model")
		
		outputPredictedKNN = knn.predict(X_test)
		st.write("Predicted Results for Testing Dataset - 1NN:")
		outputPredictedKNN
		outputPredictedKNN1 = knn1.predict(X_test)
		st.write("Predicted Results for Testing Dataset - 10NN:")
		outputPredictedKNN1
		outputPredictedKNN2 = knn2.predict(X_test)
		st.write("Predicted Results for Testing Dataset - 20NN:")
		outputPredictedKNN2
		outputPredictedKNN3 = knn3.predict(X_test)
		st.write("Predicted Results for Testing Dataset - 3NN:")
		outputPredictedKNN3

		accKNN = accuracy_score (outputPredictedKNN, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 1: ", accKNN.round(decimals = 2))
		accKNN1 = accuracy_score (outputPredictedKNN1, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 10: ", accKNN1.round(decimals = 2))
		accKNN2 = accuracy_score (outputPredictedKNN2, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 20: ", accKNN2.round(decimals = 2))
		accKNN3 = accuracy_score (outputPredictedKNN3, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 30: ", accKNN3.round(decimals = 2))

elif selectDataset == "Marijuana Crime Related":
	
	data = pd.read_csv ('crime_marijuana.csv')
	data

	st.subheader("Crime Related To Marijuana")
	data_input = data.drop(columns = ['INCIDENT_ID','FIRST_OCCURENCE_DATE','LAST_OCCURENCE_DATE','REPORTDATE', 'INCIDENT_ADDRESS', 'GEO_X', 'GEO_Y', 'OFFENSE_TYPE_ID', 'OFFENSE_CATEGORY_ID', 'MJ_RELATION_TYPE', 'NEIGHBORHOOD_ID'])
	data_input

	st.subheader("Data Target District Prediction")
	data_target = data['DISTRICT_ID']
	data_target

	st.subheader ("Training and Testing Data will be divided using Train_Test_Split")
	input_train, input_test, target_train, target_test = train_test_split (data_input, data_target, test_size = 0.2)


	st.subheader("Training data for input and target")
	st.write("Training Data Input")
	input_train
	st.write("Training Data Target")
	target_train

	st.subheader("Testing data for input and target")
	st.write("Testing Data Input")
	input_test
	st.write("Testing Data Target")
	target_test

	selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model", "K-Nearest Neighbors", "Random Forest","Support Vector Machine"])

	if selectModel == "Support Vector Machine":

		st.subheader("Support Vector Machine Model")


		st.write("Training the Model...")
		
		regressionLinear = SVR (kernel = 'linear')
		regressionLinear.fit(input_train, target_train)
		predicted_output_linear = regressionLinear.predict (input_test)
		predicted_output_linear
		mseLinear = mean_squared_error (predicted_output_linear, target_test)
		mseLinear
		st.write("The Mean Squared Error Produced by kernel = linear: ", mseLinear.round(decimals=4))

		regressionRBF = SVR (kernel = 'rbf')
		regressionRBF.fit(input_train, target_train)
		predicted_output_RBF = regressionRBF.predict (input_test)
		predicted_output_RBF
		mseRBF = mean_squared_error (predicted_output_RBF, target_test)
		mseRBF
		st.write("The Mean Squared Error Produced by kernel = rbf: ", mseRBF.round(decimals=4))

		poly_predict = SVR (kernel = 'poly')
		poly_predict.fit(input_train, target_train)
		predicted_output_poly=poly_predict.predict(input_test)
		predicted_output_poly
		msePoly = mean_squared_error(predicted_output_poly,target_test)
		msePoly
		st.write("The Mean Squared Error Produced by kernel = poly: ", msePoly.round(decimals=4))

		sigmoid_prediction=SVR(kernel='sigmoid')
		sigmoid_prediction.fit(input_train,target_train)
		predicted_output_sigmoid=sigmoid_prediction.predict(input_test)
		predicted_output_sigmoid
		mseSigmoid=mean_squared_error(predicted_output_sigmoid,target_test)
		mseSigmoid
		st.write("The Mean Squared Error Produced by kernel = sigmoid: ", mseSigmoid.round(decimals=4))

	elif selectModel == "K-Nearest Neighbors":

		st.subheader("K-Nearest Neighbors Age Estimation Model")

		knn1 = KNeighborsRegressor(n_neighbors = 1)
        
		st.write("Training the Model...")
		knn1.fit(input_train,target_train)
        
		st.write("Successfully Train the Model")
        
		target_train
		outputPredicted1 = knn1.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted1
        
		MSE1 = mean_squared_error (outputPredicted1, target_test)
		st.write("The Mean Squared Error Produced by n_neighbors = 1: ", MSE1.round(decimals=4))
        
		knn10 = KNeighborsRegressor(n_neighbors = 10)
        
		st.write("Training the Model...")       
		knn10.fit(input_train,target_train)
        
		st.write("Successfully Train the Model")
        
		target_train
		outputPredicted10 = knn10.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted10
        
		MSE10 = mean_squared_error (outputPredicted1, target_test)
		st.write("The Mean Squared Error Produced by n_neighbors = 10: ", MSE10.round(decimals=4))
        
		knn20 = KNeighborsRegressor(n_neighbors = 20)
        
		st.write("Training the Model...")
		knn20.fit(input_train,target_train)
        
		st.write("Successfully Train the Model")
        
		target_train
		outputPredicted20 = knn20.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted20
        
		MSE20 = mean_squared_error (outputPredicted20, target_test)
		st.write("The Mean Squared Error Produced by n_neighbors = 20: ", MSE20.round(decimals=4))
        
		knn30 = KNeighborsRegressor(n_neighbors = 30)
        
		st.write("Training the Model...")
		knn30.fit(input_train,target_train)
        
		st.write("Successfully Train the Model")
        
		target_train
		outputPredicted30 = knn1.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted30
        
		MSE30 = mean_squared_error (outputPredicted30, target_test)
		st.write("The Mean Squared Error Produced by n_neighbors = 30: ", MSE30.round(decimals=4))

	elif selectModel == "Random Forest":

		st.subheader("Random Forest District Estimation Model")
		
		input_train
        
		rf = RandomForestRegressor (n_estimators = 50, random_state = 0)
		st.write("Training the Model...")
		rf.fit (input_train, target_train)
		st.write("Successfully Train the Model")
		target_train

		outputPredicted50 = rf.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted50
		MSE50 = mean_squared_error (outputPredicted50, target_test)
		st.write("The Mean Squared Error Produced by n_estimator = 50: ", MSE50.round(decimals=4))
        
		rf100 = RandomForestRegressor (n_estimators = 100, random_state = 0)
		st.write("Training the Model...")
		rf100.fit (input_train, target_train)       
		st.write("Successfully Train the Model")       
		target_train
		
		outputPredicted100 = rf100.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted100       
		MSE100 = mean_squared_error (outputPredicted100, target_test)
		st.write("The Mean Squared Error Produced by n_estimator = 100: ", MSE100.round(decimals=4))

		rf200 = RandomForestRegressor (n_estimators = 200, random_state = 0)        
		st.write("Training the Model...")        
		rf200.fit (input_train, target_train)        
		st.write("Successfully Train the Model")       
		target_train
		
		outputPredicted200 = rf200.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted200
        
		MSE200 = mean_squared_error (outputPredicted50, target_test)
		st.write("The Mean Squared Error Produced by n_estimator = 200: ", MSE200.round(decimals=4))    
        
		rf300 = RandomForestRegressor (n_estimators = 300, random_state = 0)       
		st.write("Training the Model...")
		rf300.fit (input_train, target_train)       
		st.write("Successfully Train the Model")        
		target_train
		
		outputPredicted300 = rf300.predict(input_test)
		st.write("Predicted Results for Testing Dataset:")
		outputPredicted300        
		MSE300 = mean_squared_error (outputPredicted50, target_test)
		st.write("The Mean Squared Error Produced by n_estimator = 300: ", MSE300.round(decimals=4))

elif selectDataset == "Conclusion":

		data1 = pd.read_csv ('Berlin_Crimes_Dataset_.csv')
		#data1
		data_input = data1.drop(columns = ['Unnamed: 0', 'Year','DistrictNew', 	'Code', 'LocationNew', 'Local'])
		#data_input
		data_target = data1['DistrictNew']
		 #data_target
		input_train,input_test,target_train, target_test = train_test_split	(data_input, data_target, test_size=0.2)
		#input_train
		#input_test
		st.write("Dataset 1: ")
    	#SVM
		st.write("SVM Result: ")	
		svm1 = SVC (kernel = 'linear')
		svm2 = SVC (kernel = 'rbf')
		svm3 = SVC (kernel = 'poly')
		svm4 = SVC (kernel = 'sigmoid')

		svm1.fit (input_train, target_train)
		svm2.fit (input_train, target_train)
		svm3.fit (input_train, target_train)
		svm4.fit (input_train, target_train)

		outputPredictedlinear = svm1.predict(input_test)
		outputPredictedrbf = svm2.predict(input_test)
		outputPredictedpoly = svm3.predict(input_test)
		outputPredictedsig = svm4.predict(input_test)
		#outputPredictedlinear
		#outputPredictedrbf
		#outputPredictedpoly
		#outputPredictedsig

		acclinear = accuracy_score (outputPredictedlinear, target_test)
		accrbf = accuracy_score (outputPredictedrbf, target_test)
		accpoly = accuracy_score (outputPredictedpoly, target_test)
		accsig = accuracy_score (outputPredictedsig, target_test)
		st.write("The Accuracy Score Produced by Linear kernel: ", acclinear.round(decimals = 4))
		st.write("The Accuracy Score Produced by RBF kernel: ", accrbf.round(decimals = 4))
		st.write("The Accuracy Score Produced by Polynomial kernel: ", accpoly.round(decimals = 4))
		st.write("The Accuracy Score Produced by Sigmoid kernel: ", accsig.round(decimals = 4))
        #maximum SVM
		MaxSVM= max(acclinear,accrbf,accpoly,accsig)
		
		MAX1=""

		if MaxSVM == acclinear :
			MAX1="Linear kernel"
		elif MaxSVM == accrbf :
			MAX1="RBF kernel"
		elif MaxSVM == accpoly :
			MAX1 = "Polynomial kernel"
		else : 
			MAX1 = "Sigmoid kernel"	
		st.write("Maximum For SVM produced by ",MAX1," = ",MaxSVM)
		st.write("_____________________________________________________________________________")

  	   #KNN
		st.write("KNN Result")
		knn = KNeighborsClassifier (n_neighbors = 1)
		knn1 = KNeighborsClassifier (n_neighbors = 10)
		knn2 = KNeighborsClassifier (n_neighbors = 20)
		knn3 = KNeighborsClassifier (n_neighbors = 30)

		knn.fit (input_train, target_train)
		knn1.fit (input_train, target_train)
		knn2.fit (input_train, target_train)
		knn3.fit (input_train, target_train)

		outputPredictedKNN = knn.predict(input_test)
		#outputPredictedKNN
		outputPredictedKNN1 = knn1.predict(input_test)
		#outputPredictedKNN1
		outputPredictedKNN2 = knn2.predict(input_test)
	 	#outputPredictedKNN2
		outputPredictedKNN3 = knn3.predict(input_test)
	 	#outputPredictedKNN3
		
		accKNN = accuracy_score (outputPredictedKNN, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 1: ", accKNN.round(decimals = 4))
		accKNN1 = accuracy_score (outputPredictedKNN1, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 10: ", accKNN1.round(decimals = 4))
		accKNN2 = accuracy_score (outputPredictedKNN2, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 20: ", accKNN2.round(decimals = 4))
		accKNN3 = accuracy_score (outputPredictedKNN3, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 30: ", accKNN3.round(decimals = 4))
		#maximum KNN
		MaxKNN= max(accKNN,accKNN1,accKNN2,accKNN3)
		
		MAX2=""
		if MaxKNN == accKNN :
			MAX2="Number of Nearest Neighbors 1"
		elif MaxKNN == accKNN1 :
			MAX2="Number of Nearest Neighbors 10"
		elif MaxKNN == accKNN2 :
			MAX2 = "Number of Nearest Neighbors 20"
		else : 
			MAX2 = "Number of Nearest Neighbors 30"	
		st.write("Maximum For KNN produced by ",MAX2," = ",MaxKNN)
		st.write("_____________________________________________________________________________")

		#RANDOMFOREST
		st.write("Random Forest Result: ")
		rf = RandomForestClassifier (n_estimators = 50, random_state = 0)
		rf1 = RandomForestClassifier (n_estimators = 100, random_state = 0)
		rf2 = RandomForestClassifier (n_estimators = 200, random_state = 0)
		rf3 = RandomForestClassifier (n_estimators = 300, random_state = 0)

		rf.fit (input_train, target_train)
		rf1.fit (input_train, target_train)
		rf2.fit (input_train, target_train)
		rf3.fit (input_train, target_train)

		outputPredicted50 = rf.predict(input_test)
		#outputPredicted50
		outputPredicted100 = rf1.predict(input_test)
		#outputPredicted100
		outputPredicted200 = rf2.predict(input_test)
		#outputPredicted200
		outputPredicted300 = rf3.predict(input_test)
		#outputPredicted300

		acc50 = accuracy_score (outputPredicted50, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 50: ", acc50.round(decimals = 2))
		acc100 = accuracy_score (outputPredicted100, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 100: ", acc100.round(decimals = 2))
		acc200 = accuracy_score (outputPredicted200, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 200: ", acc200.round(decimals = 2))
		acc300 = accuracy_score (outputPredicted300, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 300: ", acc300.round(decimals = 2))

		#maximum random forest
		MaxRF=0
		MAX3=""
		MaxRF= max (acc50,acc100,acc200,acc300)  
		

		if MaxRF == acc50 :
			MAX3="n_estimator = 50"
		elif MaxRF == acc100 :
			MAX3="n_estimator = 100"
		elif MaxRF == acc200 :
			MAX3 = "n_estimator = 200"
		else : 
			MAX3 = "n_estimator = 300"	
		st.write("Maximum For Random Forest produced by ",MAX3," = ",MaxRF)
		st.write("_____________________________________________________________________________")

#______________________________________________________________________________________________________________________
		st.write("Dataset 2")
		
		data = pd.read_csv ('cromeboston.csv',encoding='cp1252') 
		#data

		def tran_dayofweek(x) :             
			if x == 'Monday':
				return 1
			if x == 'Tuesday':
				return 2
			if x == 'Wednesday':
				return 3
			if x == 'Thursday':
				return 4
			if x == 'Friday':
				return 5
			if x == 'Saturday':
				return 6
			if x == 'Sunday':
				return 7
	
	
		data['Trans_DayOfWeek'] = data['DAY_OF_WEEK'].apply(tran_dayofweek)
		data_input = data.drop(columns=['INCIDENT_NUMBER','OFFENSE_CODE_GROUP','OFFENSE_DESCRIPTION','DISTRICT','REPORTING_AREA','DAY_OF_WEEK','UCR_PART','STREET','Lat','Long'])  
		#data_input


		data_target = data['DISTRICT']
		#data_target

		input_train, input_test, target_train, target_test = train_test_split(data_input, data_target, test_size=0.2)
 	    #input_train
 	    #target_train
  	    #input_test
   	    #target_test

		svm1 = SVC (kernel = 'linear')
		svm2 = SVC (kernel = 'rbf')
		svm3 = SVC (kernel = 'poly')
		svm4 = SVC (kernel = 'sigmoid')

		svm1.fit (input_train, target_train)
		svm2.fit (input_train,target_train)
		svm3.fit (input_train, target_train)
		svm4.fit (input_train, target_train)

		outputPredictedlinear = svm1.predict(input_test)
		outputPredictedrbf = svm2.predict(input_test)
		outputPredictedpoly = svm3.predict(input_test)
		outputPredictedsig = svm4.predict(input_test)

		#outputPredictedlinear
		#outputPredictedrbf
		#outputPredictedpoly
		#outputPredictedsig

		st.write("SVM Result")
		acclinear = accuracy_score (outputPredictedlinear, target_test)
		accrbf = accuracy_score (outputPredictedrbf, target_test)
		accpoly = accuracy_score (outputPredictedpoly, target_test)
		accsig = accuracy_score (outputPredictedsig, target_test)
		st.write("The Accuracy Score Produced by Linear kernel: ", acclinear.round(decimals = 2))
		st.write("The Accuracy Score Produced by RBF kernel: ", accrbf.round(decimals = 2))
		st.write("The Accuracy Score Produced by Polynomial kernel: ", accpoly.round(decimals = 2))
		st.write("The Accuracy Score Produced by Sigmoid kernel: ", accsig.round(decimals = 2))

		#maximum SVM
		MaxSVM2= max(acclinear,accrbf,accpoly,accsig)
		
		MAX12=""

		if MaxSVM2 == acclinear :
			MAX12="Linear kernel"
		elif MaxSVM2 == accrbf :
			MAX12="RBF kernel"
		elif MaxSVM2 == accpoly :
			MAX12 = "Polynomial kernel"
		else : 
			MAX12 = "Sigmoid kernel"	
		st.write("Maximum For SVM produced by ",MAX12," = ",MaxSVM2)
		st.write("_____________________________________________________________________________")

		rf = RandomForestClassifier (n_estimators = 50, random_state = 0)
		rf1 = RandomForestClassifier (n_estimators = 100, random_state = 0)
		rf2 = RandomForestClassifier (n_estimators = 200, random_state = 0)
		rf3 = RandomForestClassifier (n_estimators = 300, random_state = 0)

		rf.fit (input_train, target_train)
		rf1.fit (input_train, target_train)
		rf2.fit (input_train, target_train)
		rf3.fit (input_train, target_train)


		outputPredicted50 = rf.predict(input_test)
		#outputPredicted50
		outputPredicted100 = rf1.predict(input_test)
		#outputPredicted100
		outputPredicted200 = rf2.predict(input_test)
		#outputPredicted200
		outputPredicted300 = rf3.predict(input_test)
		#outputPredicted300 

		st.write("Random Forest Result")
		acc50 = accuracy_score (outputPredicted50, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 50: ", acc50.round(decimals = 2))
		acc100 = accuracy_score (outputPredicted100, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 100: ", acc100.round(decimals = 2))
		acc200 = accuracy_score (outputPredicted200, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 200: ", acc200.round(decimals = 2))
		acc300 = accuracy_score (outputPredicted300, target_test)
		st.write("The Accuracy Score Produced by n_estimator = 300: ", acc300.round(decimals = 2))\

		MaxRF2=0
		MAX32=""
		MaxRF2= max (acc50,acc100,acc200,acc300)  
		

		if MaxRF2 == acc50 :
			MAX32="n_estimator = 50"
		elif MaxRF == acc100 :
			MAX32="n_estimator = 100"
		elif MaxRF2 == acc200 :
			MAX32 = "n_estimator = 200"
		else : 
			MAX32 = "n_estimator = 300"	
		st.write("Maximum For Random Forest produced by ",MAX32," = ",MaxRF2)
		st.write("_____________________________________________________________________________")

		knn = KNeighborsClassifier (n_neighbors = 1)
		knn1 = KNeighborsClassifier (n_neighbors = 10)
		knn2 = KNeighborsClassifier (n_neighbors = 20)
		knn3 = KNeighborsClassifier (n_neighbors = 30)

		knn.fit (input_train, target_train)
		knn1.fit (input_train, target_train)
		knn2.fit (input_train, target_train)
		knn3.fit (input_train, target_train)

		outputPredictedKNN = knn.predict(input_test)
		#outputPredictedKNN
		outputPredictedKNN1 = knn1.predict(input_test)
		#outputPredictedKNN1
		outputPredictedKNN2 = knn2.predict(input_test)
 	  # outputPredictedKNN2
		outputPredictedKNN3 = knn3.predict(input_test)
		#outputPredictedKNN3

		st.write("KNN Result")
		accKNN = accuracy_score (outputPredictedKNN, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 1: ", accKNN.round(decimals = 2))
		accKNN1 = accuracy_score (outputPredictedKNN1, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 10: ", accKNN1.round(decimals = 2))
		accKNN2 = accuracy_score (outputPredictedKNN2, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 20: ", accKNN2.round(decimals = 2))
		accKNN3 = accuracy_score (outputPredictedKNN3, target_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 30: ", accKNN3.round(decimals = 2))

		#maximum KNN
		MaxKNN2= max(accKNN,accKNN1,accKNN2,accKNN3)
		
		MAX22=""
		if MaxKNN2 == accKNN :
			MAX22="Number of Nearest Neighbors 1"
		elif MaxKNN2 == accKNN1 :
			MAX22="Number of Nearest Neighbors 10"
		elif MaxKNN2 == accKNN2 :
			MAX22 = "Number of Nearest Neighbors 20"
		else : 
			MAX22 = "Number of Nearest Neighbors 30"	
		st.write("Maximum For KNN produced by ",MAX22," = ",MaxKNN2)
		st.write("_____________________________________________________________________________")
		
#______________________________________________________________________________________________________________________		
		st.write("Dataset 3") 
		data_NYPD = pd.read_csv ('nypd-complaint-data-current-year-to-date1.csv')
		data_NYPD

		def tran_suspsex(x):
			if x == 'M':
				return 1
			if x == 'F':
				return 0

		data_NYPD['Trans_SuspSex'] = data_NYPD['SUSP_SEX'].apply(tran_suspsex)

		def tran_vicsex(x):
			if x == 'M':
				return 1
			if x == 'F':
				return 0

		data_NYPD['Trans_VicSex'] = data_NYPD['VIC_SEX'].apply(tran_vicsex)

		def tran_vicagegroup(x):
			if x == '<18':
				return 0
			if x == '18-24':
				return 1
			if x == '25-44':
				return 2
			if x == '45-64':
				return 3
			if x == '65+':
				return 4

		data_NYPD['Trans_VicAgeGroup'] = data_NYPD['VIC_AGE_GROUP'].apply(tran_vicagegroup)

		def tran_vicrace(x):
			if x == 'BLACK':
				return 0
			if x == 'WHITE':
				return 1
			if x == 'ASIAN/PAC.ISL':
				return 2
			if x == 'WHITE HISPANIC':
				return 3
			if x == 'BLACK HISPANIC':
				return 4
			if x == 'AMER IND':
				return 5

		data_NYPD['Trans_VicRace'] = data_NYPD['VIC_RACE'].apply(tran_vicrace)

		def tran_susprace(x):
			if x == 'BLACK':
				return 0
			if x == 'WHITE':
				return 1
			if x == 'ASIAN/PAC.ISL':
				return 2
			if x == 'WHITE HISPANIC':
				return 3
			if x == 'BLACK HISPANIC':
				return 4
			if x == 'AMER IND':
				return 5

		data_NYPD['Trans_SuspRace'] = data_NYPD['SUSP_RACE'].apply(tran_susprace)

		def tran_suspagegroup(x):
			if x == '<18':
				return 0
			if x == '18-24':
				return 1
			if x == '25-44':
				return 2
			if x == '45-64':
				return 3
			if x == '65+':
				return 4
    
		data_NYPD['Trans_SuspAgeGroup'] = data_NYPD['SUSP_AGE_GROUP'].apply(tran_suspagegroup)

		data_input = data_NYPD.drop(columns = ['CMPLNT_NUM','ADDR_PCT_CD','BORO_NM','CMPLNT_FR_DT','CMPLNT_FR_TM','CMPLNT_TO_DT','CMPLNT_TO_TM',
                              	 'CRM_ATPT_CPTD_CD','HADEVELOPT','HOUSING_PSA','JURISDICTION_CODE','JURIS_DESC','KY_CD','LAW_CAT_CD',
                             	  'LOC_OF_OCCUR_DESC','OFNS_DESC','PARKS_NM','PATROL_BORO','PD_CD','PD_DESC','PREM_TYP_DESC','RPT_DT',
                             	  'STATION_NAME','SUSP_AGE_GROUP','SUSP_RACE','SUSP_SEX','TRANSIT_DISTRICT','VIC_AGE_GROUP','VIC_RACE',
                             	  'VIC_SEX','X_COORD_CD','Y_COORD_CD','Latitude','Longitude','Lat_Lon'])
		#data_input
		data_target = data_NYPD["OFNS_DESC"]
		#data_target

		X_train, X_test, y_train, y_test = train_test_split(data_input, data_target, test_size=0.2)

		#X_train
		#y_train
		#X_test
		#y_test

		svm1 = SVC (kernel = 'linear')
		svm2 = SVC (kernel = 'rbf')
		svm3 = SVC (kernel = 'poly')
		svm4 = SVC (kernel = 'sigmoid')

		svm1.fit (X_train, y_train)
		svm2.fit (X_train, y_train)
		svm3.fit (X_train, y_train)
		svm4.fit (X_train, y_train)

		outputPredictedlinear = svm1.predict(X_test)
		outputPredictedrbf = svm2.predict(X_test)
		outputPredictedpoly = svm3.predict(X_test)
		outputPredictedsig = svm4.predict(X_test)
		#st.write("Predicted Results for Testing Dataset - Linear :")
		#outputPredictedlinear
		#st.write("Predicted Results for Testing Dataset - RBF :")
		#outputPredictedrbf
		#st.write("Predicted Results for Testing Dataset - Polynomial :")
		#outputPredictedpoly
		#st.write("Predicted Results for Testing Dataset - Sigmoid :")
		#outputPredictedsig

		st.write("SVM")
		acclinear = accuracy_score (outputPredictedlinear, y_test)
		accrbf = accuracy_score (outputPredictedrbf, y_test)
		accpoly = accuracy_score (outputPredictedpoly, y_test)
		accsig = accuracy_score (outputPredictedsig, y_test)
		st.write("The Accuracy Score Produced by Linear kernel: ", acclinear.round(decimals = 2))
		st.write("The Accuracy Score Produced by RBF kernel: ", accrbf.round(decimals = 2))
		st.write("The Accuracy Score Produced by Polynomial kernel: ", accpoly.round(decimals = 2))
		st.write("The Accuracy Score Produced by Sigmoid kernel: ", accsig.round(decimals = 2))

		MaxSVM3= max(acclinear,accrbf,accpoly,accsig)
		
		MAX13=""

		if MaxSVM3 == acclinear :
			MAX13="Linear kernel"
		elif MaxSVM3 == accrbf :
			MAX13="RBF kernel"
		elif MaxSVM3 == accpoly :
			MAX13 = "Polynomial kernel"
		else : 
			MAX13 = "Sigmoid kernel"	
		st.write("Maximum For SVM produced by ",MAX13," = ",MaxSVM3)
		st.write("_____________________________________________________________________________")

		rf = RandomForestClassifier (n_estimators = 50, random_state = 0)
		rf1 = RandomForestClassifier (n_estimators = 100, random_state = 0)
		rf2 = RandomForestClassifier (n_estimators = 200, random_state = 0)
		rf3 = RandomForestClassifier (n_estimators = 300, random_state = 0)

		st.write("Training the Model...")
		rf.fit (X_train, y_train)
		rf1.fit (X_train, y_train)
		rf2.fit (X_train, y_train)
		rf3.fit (X_train, y_train)
		
		outputPredicted50 = rf.predict(X_test)
		#outputPredicted50
		outputPredicted100 = rf1.predict(X_test)
		#outputPredicted100
		outputPredicted200 = rf2.predict(X_test)
		#outputPredicted200
		outputPredicted300 = rf3.predict(X_test)
		#outputPredicted300

		st.write("Random Forest")
		acc50 = accuracy_score (outputPredicted50, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 50: ", acc50.round(decimals = 2))
		acc100 = accuracy_score (outputPredicted100, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 100: ", acc100.round(decimals = 2))
		acc200 = accuracy_score (outputPredicted200, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 200: ", acc200.round(decimals = 2))
		acc300 = accuracy_score (outputPredicted300, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 300: ", acc300.round(decimals = 2))

		#maximum random forest
		MaxRF3=0
		MAX33=""
		MaxRF3= max (acc50,acc100,acc200,acc300)  
		

		if MaxRF3 == acc50 :
			MAX33="n_estimator = 50"
		elif MaxRF3 == acc100 :
			MAX33="n_estimator = 100"
		elif MaxRF3 == acc200 :
			MAX33 = "n_estimator = 200"
		else : 
			MAX33 = "n_estimator = 300"	
		st.write("Maximum For Random Forest produced by ",MAX33," = ",MaxRF3)
		st.write("_____________________________________________________________________________")

		knn = KNeighborsClassifier (n_neighbors = 1)
		knn1 = KNeighborsClassifier (n_neighbors = 10)
		knn2 = KNeighborsClassifier (n_neighbors = 20)
		knn3 = KNeighborsClassifier (n_neighbors = 30)

		#st.write("Training the Model...")
		knn.fit (X_train, y_train)
		knn1.fit (X_train, y_train)
		knn2.fit (X_train, y_train)
		knn3.fit (X_train, y_train)

		outputPredictedKNN = knn.predict(X_test)
		#st.write("Predicted Results for Testing Dataset - 1NN:")
		#outputPredictedKNN
		outputPredictedKNN1 = knn1.predict(X_test)
		#st.write("Predicted Results for Testing Dataset - 10NN:")
		#outputPredictedKNN1
		outputPredictedKNN2 = knn2.predict(X_test)
		#st.write("Predicted Results for Testing Dataset - 20NN:")
		#outputPredictedKNN2
		outputPredictedKNN3 = knn3.predict(X_test)
		#st.write("Predicted Results for Testing Dataset - 3NN:")
		#outputPredictedKNN3

		st.write("KNN")
		accKNN = accuracy_score (outputPredictedKNN, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 1: ", accKNN.round(decimals = 2))
		accKNN1 = accuracy_score (outputPredictedKNN1, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 10: ", accKNN1.round(decimals = 2))
		accKNN2 = accuracy_score (outputPredictedKNN2, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 20: ", accKNN2.round(decimals = 2))
		accKNN3 = accuracy_score (outputPredictedKNN3, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 30: ", accKNN3.round(decimals = 2))

		#maximum KNN
		MaxKNN3= max(accKNN,accKNN1,accKNN2,accKNN3)
		
		MAX23=""
		if MaxKNN3 == accKNN :
			MAX23="Number of Nearest Neighbors 1"
		elif MaxKNN3 == accKNN1 :
			MAX23="Number of Nearest Neighbors 10"
		elif MaxKNN3 == accKNN2 :
			MAX23 = "Number of Nearest Neighbors 20"
		else : 
			MAX23 = "Number of Nearest Neighbors 30"	
		st.write("Maximum For KNN produced by ",MAX23," = ",MaxKNN3)
		st.write("_____________________________________________________________________________")
#____________________________________________________________________________________________________________________________________
		st.write("Dataset 4")
		data_crimeindia = pd.read_csv ('crime_under_women_streamlit.csv')
		#data_crimeindia
    
		data_inputindia = data_crimeindia.drop(columns = ['Area_Name','Year','Group_Name','Sub_Group_Name']) 
		#data_inputindia

		data_targetindia = data_crimeindia['Year'] 
		#data_targetindia

		X_train, X_test, y_train, y_test = train_test_split(data_inputindia, data_targetindia, test_size=0.2)

		#X_train
		#y_train
		#X_test
		#y_test

		svm1 = SVC (kernel = 'linear')
		svm2 = SVC (kernel = 'rbf')
		svm3 = SVC (kernel = 'poly')
		svm4 = SVC (kernel = 'sigmoid')

		svm1.fit (X_train, y_train)
		svm2.fit (X_train, y_train)
		svm3.fit (X_train, y_train)
		svm4.fit (X_train, y_train)
		
		outputPredictedlinear = svm1.predict(X_test)
		outputPredictedrbf = svm2.predict(X_test)
		outputPredictedpoly = svm3.predict(X_test)
		outputPredictedsig = svm4.predict(X_test)
		#st.write("Predicted Results for Testing Dataset - Linear :")
		#outputPredictedlinear
		#st.write("Predicted Results for Testing Dataset - RBF :")
		#outputPredictedrbf
		#st.write("Predicted Results for Testing Dataset - Polynomial :")
		#outputPredictedpoly
		#st.write("Predicted Results for Testing Dataset - Sigmoid :")
		#outputPredictedsig

		acclinear = accuracy_score (outputPredictedlinear, y_test)
		accrbf = accuracy_score (outputPredictedrbf, y_test)
		accpoly = accuracy_score (outputPredictedpoly, y_test)
		accsig = accuracy_score (outputPredictedsig, y_test)
		st.write("The Accuracy Score Produced by Linear kernel: ", acclinear.round(decimals = 2))
		st.write("The Accuracy Score Produced by RBF kernel: ", accrbf.round(decimals = 2))
		st.write("The Accuracy Score Produced by Polynomial kernel: ", accpoly.round(decimals = 2))
		st.write("The Accuracy Score Produced by Sigmoid kernel: ", accsig.round(decimals = 2))

		MaxSVM4= max(acclinear,accrbf,accpoly,accsig)
		
		MAX14=""

		if MaxSVM4 == acclinear :
			MAX14="Linear kernel"
		elif MaxSVM4 == accrbf :
			MAX14="RBF kernel"
		elif MaxSVM4 == accpoly :
			MAX14 = "Polynomial kernel"
		else : 
			MAX14 = "Sigmoid kernel"	
		st.write("Maximum For SVM produced by ",MAX14," = ",MaxSVM4)
		st.write("_____________________________________________________________________________")

		rf = RandomForestClassifier (n_estimators = 50, random_state = 0)
		rf1 = RandomForestClassifier (n_estimators = 100, random_state = 0)
		rf2 = RandomForestClassifier (n_estimators = 200, random_state = 0) 
		rf3 = RandomForestClassifier (n_estimators = 300, random_state = 0)

		rf.fit (X_train, y_train)
		rf1.fit (X_train, y_train)
		rf2.fit (X_train, y_train)
		rf3.fit (X_train, y_train)
	
		outputPredicted50 = rf.predict(X_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted50
		outputPredicted100 = rf1.predict(X_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted100
		outputPredicted200 = rf2.predict(X_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted200
		outputPredicted300 = rf3.predict(X_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted300

		acc50 = accuracy_score (outputPredicted50, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 50: ", acc50.round(decimals = 2))
		acc100 = accuracy_score (outputPredicted100, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 100: ", acc100.round(decimals = 2))
		acc200 = accuracy_score (outputPredicted200, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 200: ", acc200.round(decimals = 2))
		acc300 = accuracy_score (outputPredicted300, y_test)
		st.write("The Accuracy Score Produced by n_estimator = 300: ", acc300.round(decimals = 2))

		#maximum random forest
		MaxRF4=0
		MAX34=""
		MaxRF4= max (acc50,acc100,acc200,acc300)  
		

		if MaxRF4 == acc50 :
			MAX34="n_estimator = 50"
		elif MaxRF4 == acc100 :
			MAX34="n_estimator = 100"
		elif MaxRF4 == acc200 :
			MAX34 = "n_estimator = 200"
		else : 
			MAX34 = "n_estimator = 300"	
		st.write("Maximum For Random Forest produced by ",MAX34," = ",MaxRF4)
		st.write("_____________________________________________________________________________")
	
		knn = KNeighborsClassifier (n_neighbors = 1)
		knn1 = KNeighborsClassifier (n_neighbors = 10)
		knn2 = KNeighborsClassifier (n_neighbors = 20)
		knn3 = KNeighborsClassifier (n_neighbors = 30)

		knn.fit (X_train, y_train)
		knn1.fit (X_train, y_train)
		knn2.fit (X_train, y_train)
		knn3.fit (X_train, y_train)
	
		outputPredictedKNN = knn.predict(X_test)
		#st.write("Predicted Results for Testing Dataset - 1NN:")
		#outputPredictedKNN
		outputPredictedKNN1 = knn1.predict(X_test)
		#st.write("Predicted Results for Testing Dataset - 10NN:")
		#outputPredictedKNN1
		outputPredictedKNN2 = knn2.predict(X_test)
		#st.write("Predicted Results for Testing Dataset - 20NN:")
		#outputPredictedKNN2
		outputPredictedKNN3 = knn3.predict(X_test)
		#st.write("Predicted Results for Testing Dataset - 3NN:")
		#outputPredictedKNN3

		accKNN = accuracy_score (outputPredictedKNN, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 1: ", accKNN.round(decimals = 2))
		accKNN1 = accuracy_score (outputPredictedKNN1, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 10: ", accKNN1.round(decimals = 2))
		accKNN2 = accuracy_score (outputPredictedKNN2, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 20: ", accKNN2.round(decimals = 2))
		accKNN3 = accuracy_score (outputPredictedKNN3, y_test)
		st.write("The Accuracy Score Produced by KNN with number of nearest neighbors 30: ", accKNN3.round(decimals = 2))

		#maximum KNN
		MaxKNN4= max(accKNN,accKNN1,accKNN2,accKNN3)
		
		MAX24=""
		if MaxKNN4 == accKNN :
			MAX24="Number of Nearest Neighbors 1"
		elif MaxKNN4 == accKNN1 :
			MAX24="Number of Nearest Neighbors 10"
		elif MaxKNN4 == accKNN2 :
			MAX24 = "Number of Nearest Neighbors 20"
		else : 
			MAX24 = "Number of Nearest Neighbors 30"	
		st.write("Maximum For KNN produced by ",MAX24," = ",MaxKNN4)
		st.write("_____________________________________________________________________________")

	#_______________________________________________________________________________________________________________________________________	
	
		data = pd.read_csv ('crime_marijuana.csv')
		#data
		data_input = data.drop(columns = ['INCIDENT_ID','FIRST_OCCURENCE_DATE','LAST_OCCURENCE_DATE','REPORTDATE', 'INCIDENT_ADDRESS', 'GEO_X', 'GEO_Y', 'OFFENSE_TYPE_ID', 'OFFENSE_CATEGORY_ID', 'MJ_RELATION_TYPE', 'NEIGHBORHOOD_ID'])
		#data_input
		data_target = data['DISTRICT_ID']
		#data_target
		input_train, input_test, target_train, target_test = train_test_split (data_input, data_target, test_size = 0.2)
 		
		st.write("SVM Result")      	
		regressionLinear = SVR (kernel = 'linear')
		regressionLinear.fit(input_train, target_train)  
		predicted_output_linear = regressionLinear.predict (input_test) 
		#predicted_output_linear
		mseLinear = mean_squared_error (predicted_output_linear, target_test)
		#mseLinear
		st.write("The Mean Squared Error Produced by kernel = linear: ", mseLinear.round(decimals=4))

		regressionRBF = SVR (kernel = 'rbf')
		regressionRBF.fit(input_train, target_train)
		predicted_output_RBF = regressionRBF.predict (input_test)
		#predicted_output_RBF
		mseRBF = mean_squared_error (predicted_output_RBF, target_test)
		#mseRBF
		st.write("The Mean Squared Error Produced by kernel = rbf: ", mseRBF.round(decimals=4))

		poly_predict = SVR (kernel = 'poly')
		poly_predict.fit(input_train, target_train)
		predicted_output_poly=poly_predict.predict(input_test)
		#predicted_output_poly
		msePoly = mean_squared_error(predicted_output_poly,target_test)
		#msePoly
		st.write("The Mean Squared Error Produced by kernel = poly: ", msePoly.round(decimals=4))

		sigmoid_prediction=SVR(kernel='sigmoid')
		sigmoid_prediction.fit(input_train,target_train)
		predicted_output_sigmoid=sigmoid_prediction.predict(input_test)
		#predicted_output_sigmoid
		mseSigmoid=mean_squared_error(predicted_output_sigmoid,target_test)
		#mseSigmoid
		st.write("The Mean Squared Error Produced by kernel = sigmoid: ", mseSigmoid.round(decimals=4))

		#min KNN
		MinSVM= max(mseLinear,mseRBF,msePoly,mseSigmoid)
		
		Min1=""
		if MinSVM == mseLinear :
			Min1="Linear Kernel"
		elif MinSVM == mseRBF:
			Min1="RBF Kernel"
		elif MinSVM == msePoly :
			Min1 = "Polynomial Kernel"
		else : 
			Min1 = "Sigmoid Kernel"	
		st.write("Minimum For KNN produced by ",Min1," = ",MinSVM)
		st.write("_____________________________________________________________________________")
		
		st.write("KNN Result")
		knn1 = KNeighborsRegressor(n_neighbors = 1)

		#st.write("Training the Model...")
		knn1.fit(input_train,target_train) 
		#st.write("Successfully Train the Model")
		#target_train
		outputPredicted1 = knn1.predict(input_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted1
    
		MSE1 = mean_squared_error (outputPredicted1, target_test)
		st.write("The Mean Squared Error Produced by n_neighbors = 1: ", MSE1.round(decimals=4))
        
		knn10 = KNeighborsRegressor(n_neighbors = 10)
        
		#st.write("Training the Model...")       
		knn10.fit(input_train,target_train)
		#st.write("Successfully Train the Model")
		#target_train
		outputPredicted10 = knn10.predict(input_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted10
        
		MSE10 = mean_squared_error (outputPredicted1, target_test)
		st.write("The Mean Squared Error Produced by n_neighbors = 10: ", MSE10.round(decimals=4))
        
		knn20 = KNeighborsRegressor(n_neighbors = 20)      
		#st.write("Training the Model...")
		knn20.fit(input_train,target_train)       
		#st.write("Successfully Train the Model")       
		#target_train
		outputPredicted20 = knn20.predict(input_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted20
        
		MSE20 = mean_squared_error (outputPredicted20, target_test)
		st.write("The Mean Squared Error Produced by n_neighbors = 20: ", MSE20.round(decimals=4))
        
		knn30 = KNeighborsRegressor(n_neighbors = 30)
		#st.write("Training the Model...")
		knn30.fit(input_train,target_train)       
		#st.write("Successfully Train the Model")        
		#target_train
		outputPredicted30 = knn1.predict(input_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted30
        
		MSE30 = mean_squared_error (outputPredicted30, target_test)
		st.write("The Mean Squared Error Produced by n_neighbors = 30: ", MSE30.round(decimals=4))

		#min KNN
		MinKNN= max(MSE1,MSE10,MSE20,MSE30)
		
		Min2=""
		if MinKNN == MSE1 :
			Min2="Number of Nearest Neighbors 1"
		elif MinKNN == MSE10 :
			Min2="Number of Nearest Neighbors 10"
		elif MinKNN == MSE20 :
			Min2 = "Number of Nearest Neighbors 20"
		else : 
			Min2 = "Number of Nearest Neighbors 30"	
		st.write("Minimum For KNN produced by ",Min2," = ",MinKNN)
		st.write("_____________________________________________________________________________")
		
    #Random Forest
		st.write("Random Forest Result")
		#input_train
		rf = RandomForestRegressor (n_estimators = 50, random_state = 0)
		#st.write("Training the Model...")
		rf.fit (input_train, target_train)
		#st.write("Successfully Train the Model")
		#target_train 

		outputPredicted50 = rf.predict(input_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted50
		MSE50 = mean_squared_error (outputPredicted50, target_test)
		st.write("The Mean Squared Error Produced by n_estimator = 50: ", MSE50.round(decimals=4))
        
		rf100 = RandomForestRegressor (n_estimators = 100, random_state = 0)
		#st.write("Training the Model...")
		rf100.fit (input_train, target_train)       
		#st.write("Successfully Train the Model")       
		#target_train
		
		outputPredicted100 = rf100.predict(input_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted100       
		MSE100 = mean_squared_error (outputPredicted100, target_test)
		st.write("The Mean Squared Error Produced by n_estimator = 100: ", MSE100.round(decimals=4))

		rf200 = RandomForestRegressor (n_estimators = 200, random_state = 0)        
		#st.write("Training the Model...")        
		rf200.fit (input_train, target_train)        
		#st.write("Successfully Train the Model")       
		#target_train
		
		outputPredicted200 = rf200.predict(input_test)
		#st.write("Predicted Results for Testing Dataset:")
		#outputPredicted200
        
		MSE200 = mean_squared_error (outputPredicted50, target_test)
		st.write("The Mean Squared Error Produced by n_estimator = 200: ", MSE200.round(decimals=4))    
        
		rf300 = RandomForestRegressor (n_estimators = 300, random_state = 0)       
		#st.write("Training the Model...")
		rf300.fit (input_train, target_train)       
		#st.write("Successfully Train the Model")        
		#target_train
		
		outputPredicted300 = rf300.predict(input_test)
		#t.write("Predicted Results for Testing Dataset:")
		#outputPredicted300        
		MSE300 = mean_squared_error (outputPredicted50, target_test)
		st.write("The Mean Squared Error Produced by n_estimator = 300: ", MSE300.round(decimals=4))

		#maximum random forest
		MinRF=0
		Min3=""
		MinRF= max (MSE50,MSE100,MSE200,MSE300)  
		

		if MinRF == MSE50 :
			Min3="n_estimator = 50"
		elif MinRF == MSE100 :
			Min3="n_estimator = 100"
		elif MinRF == MSE200 :
			Min3 = "n_estimator = 200"
		else : 
			Min3 = "n_estimator = 300"	
		st.write("Minimum For Random Forest produced by ",Min3," = ",MinRF)
		st.write("_____________________________________________________________________________")