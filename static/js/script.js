

// Sample match data for live match display


  fetch('{{ url_for('static', filename='json/Matchdetails.json') }}')

    .then(response => response.json())  // Parse the JSON data
    .then(data => {
        // Assuming the JSON has a key "items" that is an array
        const displayElement = document.getElementById('jsonData');
        displayElement.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    })
    .catch(error => {
        console.error('Error fetching JSON:', error);
    });



// JS Script for predict page
    async function predict() {
      const team1 = document.getElementById('team1').value;
      const team2 = document.getElementById('team2').value;
  
      if (team1 === team2) {
          alert("Please select two different teams.");
          return;
      }
  
      const response = await fetch('/predict', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify({ team1, team2 })
      });
  
      if (response.ok) {
          const probabilities = await response.json();
          document.getElementById('result').innerHTML = `
              <h2>Prediction Result:</h2>
              <p>${team1}: ${probabilities[team1]}%</p>
              <p>${team2}: ${probabilities[team2]}%</p>
          `;
      } else {
          alert('Error making prediction. Please try again.');
      }
  }

  card.addEventListener("click", () => {
    localStorage.setItem('matchDetails', JSON.stringify(match));
    window.location.href = "predict.html";  // This needs to be updated
  });
  
  