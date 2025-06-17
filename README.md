# CourtVision_AI
CourtVision AI: NBA Game Prediction Web Application
 
CourtVision AI is a full-stack web application that leverages machine learning to predict the outcomes of National Basketball Association (NBA) games with high accuracy. Utilizing a Random Forest Classifier trained on comprehensive NBA statistical data, the system achieves a 93% prediction accuracy, as demonstrated by its correct forecast of a New York Knicks victory over the Boston Celtics on May 6, 2025. Built with a Django backend and a React.js frontend, CourtVision AI provides an intuitive interface for users to select teams and view detailed game predictions, making it a valuable tool for sports analysts, NBA teams, media outlets, and fans.
This project was developed by Divine Jacob as part of a Bachelor of Science dissertation at the University of Lincoln, submitted in May 2025, under the supervision of Heriberto Cuayahuitl Portilla.
Table of Contents

Features
Technologies
Installation
Usage
Project Structure
Data Sources
Machine Learning Model
Ethical Considerations
Limitations and Future Work
Contributing
License
Acknowledgments
Contact

Features

Accurate Predictions: Achieves 93% accuracy using a Random Forest Classifier to predict NBA game outcomes based on key statistics like field goal percentages, rebounds, and turnovers.
User-Friendly Interface: A React.js frontend allows users to select home and away teams from a dropdown of all 30 NBA teams, with error handling to prevent invalid inputs.
Clear Visualizations: Displays win probabilities, team advantages, and analytical comments in a visually appealing format.
Robust Backend: Django-powered backend with an API layer for seamless integration between the machine learning model and frontend.
Ethical Data Practices: Adheres to ethical web scraping guidelines when sourcing data from NBA.com, Basketball-Reference, and Sportsradar.
Bias Mitigation: Implements strategies to reduce data and algorithm biases, ensuring fair predictions.

Technologies

Backend:
Python 3.9+
Django 4.x
scikit-learn (for machine learning)
BeautifulSoup (for web scraping)


Frontend:
React.js 18.x
Axios (for API requests)
Tailwind CSS (for styling)


Other Tools:
Git/GitHub (version control)
PostgreSQL (database, optional for production)
Jupyter Notebook (model development and testing)
