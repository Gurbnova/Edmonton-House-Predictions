from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        'locations' : util.get_location_name()
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    location = request.form['location']
    bedroom = int(request.form['bedroom'])
    bath = int(request.form['bath'])
    sqft = float(request.form['sqft'])

    response = jsonify({
        'estimated_price' : util.get_estimated_price(location, bedroom, bath, sqft)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run()