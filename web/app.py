from flask import Flask, render_template, request

app = Flask(__name__)
RECENT = []

@app.route('/')
def index():
    return render_template('index.html', events=list(reversed(RECENT[-50:])))

@app.route('/alert', methods=['POST'])
def alert():
    try:
        data = request.get_json(force=True)
        RECENT.append(data)
        return {'ok': True}
    except Exception as e:
        return {'ok': False, 'error': str(e)}, 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
