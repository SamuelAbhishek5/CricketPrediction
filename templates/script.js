document.addEventListener('DOMContentLoaded', function () {
    // Sample match data for live match display
    const liveMatches = [
      {
        team1: 'India',
        team2: 'Australia',
        score1: '250/6',
        score2: '240/7',
        status: 'India leads by 10 runs',
      },
      {
        team1: 'Pakistan',
        team2: 'England',
        score1: '220/5',
        score2: '225/8',
        status: 'England is chasing',
      },
      {
        team1: 'South Africa',
        team2: 'New Zealand',
        score1: '310/4',
        score2: '299/6',
        status: 'South Africa leads by 11 runs',
      },
    ];
  
    const scoreContainer = document.getElementById('score-container');
  
    function populateCards() {
      liveMatches.forEach(match => {
        const card = document.createElement('div');
        card.classList.add('score-card');
        card.innerHTML = `
          <h5>${match.team1} vs ${match.team2}</h5>
          <div class="team-score">${match.score1} - ${match.score2}</div>
          <div class="match-details">${match.status}</div>
        `;
        scoreContainer.appendChild(card);
      });
    }
  
    populateCards();
  
    // Implementing scroll behavior
    const scrollLeft = document.getElementById('scroll-left');
    const scrollRight = document.getElementById('scroll-right');
    scrollLeft.addEventListener('click', () => {
      scoreContainer.scrollBy({
        left: -250,
        behavior: 'smooth',
      });
    });
  
    scrollRight.addEventListener('click', () => {
      scoreContainer.scrollBy({
        left: 250,
        behavior: 'smooth',
      });
    });
  });
  