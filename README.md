# mod3_project

This repo contains:
* Description of the project in a PDF
* Three python starter files
* Three test files
* A starter jupyter notebook# TurkorMod3Project

BUSINESS APPROACH
* We want to create several hypothoses which test the effectiveness of players based on metrics like height, position, weight. Our findings should tell us which type of players are effective in certain roles. Our aim is to prescribe draft picks for teams in the annual NBA draft. For example, if a team is looking for a 3 point shooter, which type of player height and weight should they be looking at.

THE DATA
We will be using NBA stats that were recorded between 1998-2017(inclusive). We have information on player names, height, and several parameters which are well known within the basketballing world.

HYPOTHESIS
1. Players less than 6ft7 tall have a higher 3-point percentage than those who are 6ft7 or taller. i.e. mu_short > mu_tall
2. On average, heavier players do more fouls then the lighter ones. mu_heavy > mu_light
3. Point Guards have a higher free throw percentage than shooting guards. mu_point_guard > mu_shooting_guard
4. Across positions, blocks per game differ. mu_position = mu_other positions

GOALS
Our goal is to consult the New Orleans in the NBA 2020 Draft. After years of diminshing performances in the regular season, Dez's sports consulting firm wants to help the Pelicans achieve a playoff place

Responsibilities
Jointly retrieved data usiing API and csv file (due to API limit)
Sez is testing hypothesis 1-2
Dan is testing hypothese 3-4

Summary of files
Data cleaning py file for cleaning our data
Hypothesis testing py file for functions and all hypothesis testing 
Mod3Final is the completed jupyter notebook
