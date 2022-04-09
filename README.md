# ML-Audio-Classification
MLEnd Hums and whistle data to classify songs 


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

model = RandomForestClassifier(random_state=42)
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)


rfc1=RandomForestClassifier(random_state=42, criterion= 'gini',
 max_depth= 4,
 max_features='auto',
 n_estimators=200)

rfc1.fit(X_train, y_train)