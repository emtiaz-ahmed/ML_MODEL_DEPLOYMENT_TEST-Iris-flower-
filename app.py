import flask
import pickle
from sklearn.externals import joblib
import pandas as pd

# Use pickle to load in the pre-trained model.
# with open(f'model/iris_flowers.pkl','rb') as f:
#     model = pickle.load(f)

model = joblib.load('model/iris_flowers.pkl')


app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])

# @app.route('/')

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

       
    if flask.request.method == 'POST':
        sepal_length = flask.request.form['sepal-length']
        sepal_width = flask.request.form['sepal-width']
        petal_length = flask.request.form['petal-length']
        petal_width = flask.request.form['petal-width']


        input_variables = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                       columns=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]

        return flask.render_template('main.html',
                                     original_input={'sepal_length':sepal_length,
                                                     'sepal_width':sepal_width,
                                                     'petal-length':petal_length,
                                                     'petal-width':petal_width},
                                     result=prediction,
                                     )


if __name__ == '__main__':
    app.run()