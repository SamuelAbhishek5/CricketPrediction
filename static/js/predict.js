function showAlert(message) {
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';

    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 3000);
}

document.addEventListener('DOMContentLoaded', () => {
    fetch('/dropdown_data')
        .then(response => response.json())
        .then(data => {
            const team1Dropdown = document.getElementById('team1');
            data.team1.forEach(team => {
                const option = document.createElement('option');
                option.value = team.name;
                option.textContent = team.name;
                team1Dropdown.appendChild(option);
            });

            const team2Dropdown = document.getElementById('team2');
            data.team2.forEach(team => {
                const option = document.createElement('option');
                option.value = team.name;
                option.textContent = team.name;
                team2Dropdown.appendChild(option);
            });

            const cityDropdown = document.getElementById('city');
            data.cities.forEach(city => {
                const option = document.createElement('option');
                option.value = city;
                option.textContent = city;
                cityDropdown.appendChild(option);
            });
        })
        .catch(error => {
            console.error("Error loading data:", error);
            showAlert("Error loading dropdown data. Please try again later.");
        });
});

function updateTossOptions() {
    const team1 = document.getElementById('team1').value;
    const team2 = document.getElementById('team2').value;
    const tossWinnerSelect = document.getElementById('toss_winner');
    tossWinnerSelect.innerHTML = `<option value="" disabled selected>Select Toss Winner</option>`;

    if (team1 && team2) {
        const option1 = document.createElement('option');
        option1.value = team1;
        option1.textContent = team1;

        const option2 = document.createElement('option');
        option2.value = team2;
        option2.textContent = team2;

        tossWinnerSelect.appendChild(option1);
        tossWinnerSelect.appendChild(option2);
    }
}

async function predict() {
    const team1 = document.getElementById('team1').value;
    const team2 = document.getElementById('team2').value;
    const city = document.getElementById('city').value;
    const required_runs = document.getElementById('required_runs').value;
    const remaining_overs = document.getElementById('remaining_overs').value;
    const remaining_wickets = document.getElementById('remaining_wickets').value;
    const toss_winner = document.getElementById('toss_winner').value;
    const toss_decision = document.getElementById('toss_decision').value;
    const target_runs = document.getElementById('target_runs').value;

    if (!team1 || !team2 || team1 === team2) {
        showAlert('Please select two different teams.');
        return;
    }

    if (!city) {
        showAlert('Please select a city.');
        return;
    }

    //if (required_runs <= 0 || required_runs > target_runs) {
        //showAlert('Please enter a valid number for required runs.');
       // return;
   // }

    if (!remaining_overs || remaining_overs <= 0) {
        showAlert('Please enter a valid number for remaining overs.');
        return;
    }

    if (!remaining_wickets || remaining_wickets < 0) {
        showAlert('Please enter a valid number for remaining wickets.');
        return;
    }

    if (!target_runs || target_runs <= 0) {
        showAlert('Please enter a valid number for target runs.');
        return;
    }

    const predictionData = {
        team1: team1,
        team2: team2,
        city: city,
        required_runs: required_runs,
        remaining_overs: remaining_overs,
        remaining_wickets: remaining_wickets,
        toss_winner: toss_winner,
        toss_decision: toss_decision,
        target_runs: target_runs
    };

    try {
        const response = await fetch('/predict/result', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(predictionData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const result = data.team1_win_probability > 50
            ? `${data.team1} has a higher chance of winning!`
            : `${data.team2} has a higher chance of winning!`;

        const probability = data.team1_win_probability;
        displayPrediction(result, probability, team1, team2);

        scrollToProbabilitySection();
    } catch (error) {
        console.error('Error during prediction:', error);
        showAlert('An error occurred while making the prediction. Please try again.');
    }
}

function displayPrediction(result, probability, team1Name, team2Name) {
    const winProbabilitySection = document.getElementById('win-probability');
    const probabilityChart = document.getElementById('probability-chart');
    probabilityChart.innerHTML = '';

    winProbabilitySection.style.display = 'block';

    const chartHTML = `
        <div class="probability-bar-container">
            <div class="team-name left">${team1Name}</div>
            <div class="probability-bar-background">
                <div class="probability-bar" style="width: ${probability}%;"></div>
            </div>
            <div class="team-name right">${team2Name}</div>
        </div>
        <div class="probability-text">
            <span>${probability}%</span> vs <span>${100 - probability}%</span>
        </div>`;

    probabilityChart.innerHTML = chartHTML;

    const resultText = document.createElement("p");
    resultText.textContent = result;
    resultText.style.textAlign = "center";
    resultText.style.fontWeight = "bold";
    resultText.style.color = "#333";
    winProbabilitySection.appendChild(resultText);
}

function scrollToProbabilitySection() {
    const winProbabilitySection = document.getElementById('win-probability');
    winProbabilitySection.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

window.onload = function() {
    updateTossOptions();
};