// Funktion zum Erstellen und Kompilieren des Modells
function createAndCompileModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError', metrics: ['accuracy'] });

    return model;
}

// Modell erstellen und kompilieren
const model = createAndCompileModel();

// Trainingsdaten für die y = 2x Funktion
const values  = tf.tensor2d([[1], [2], [3], [4], [5]]);
const results = tf.tensor2d([[2], [4], [6], [8], [10]]);

// Modell trainieren
model.fit(values, results, { epochs: 1000 });

// Funktion zur Vorhersage basierend auf Benutzereingabe
function predict() {
    // Benutzereingabe abrufen
    const inputX = parseFloat(document.getElementById('inputX').value);

    // Vorhersage machen
    const inputData = tf.tensor2d([[inputX]]);
    const prediction = model.predict(inputData).dataSync();

    // Ergebnisse in das HTML-Element einfügen
    const outputContainer = document.getElementById('output-container');
    outputContainer.innerHTML = `<p>Für X = ${inputX}, ist Y = ${prediction[0]}</p>`;
}