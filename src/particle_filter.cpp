/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).	
	num_particles = 100;

	particles.resize(num_particles);

	// Create a Gaussian normal distribution for x, y and theta.
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);

	// Initialize all particles (particle positon + noise, weights -> 1)
	for (int i = 0; i < num_particles; i++){
	  particles[i].id = i;
	  particles[i].x = x + dist_x(gen);			
	  particles[i].y = y + dist_y(gen);		
	  particles[i].theta = theta + dist_theta(gen);
	  particles[i].weight = 1.0;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// Sensor Noise - Normal Gaussian distributions
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {
	  // Prediction
	  if (fabs(yaw_rate) < 1e-5) {  
	    particles[i].x += velocity * delta_t * cos(particles[i].theta) + noise_x(gen);
	    particles[i].y += velocity * delta_t * sin(particles[i].theta) + noise_y(gen);
	  } 
	  else {
	    particles[i].x += velocity / yaw_rate * ( sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + noise_x(gen);
	    particles[i].y += velocity / yaw_rate * (-cos(particles[i].theta + yaw_rate*delta_t) + cos(particles[i].theta)) + noise_y(gen);
	    particles[i].theta += yaw_rate * delta_t + noise_theta(gen);
	  }
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (int i = 0; i < observations.size(); i++) {
	  int map_id = -1;
    	  double min_distance = numeric_limits<double>::max();
	  for (int j = 0; j < predicted.size(); j++) {
	    double current_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
	    if (current_distance < min_distance) {
	      min_distance = current_distance;
	      map_id = predicted[j].id;
	    }
	  }
	  observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double std_x2 = std_x * std_x;
	double std_y2 = std_y * std_y;
	double MK = 1.0/(2.0*M_PI*std_x*std_y);

	for (int i = 0; i < num_particles; i++) {
	  // Get particle parameters
	  double particle_x = particles[i].x;
	  double particle_y = particles[i].y;
	  double particle_theta = particles[i].theta;

	  // Push only landmarks in sensor range	
	  vector<LandmarkObs> landmarks_in_range;
	  for (int j = 0; j <  map_landmarks.landmark_list.size(); j++){ 
	    float landmark_x = map_landmarks.landmark_list[j].x_f;
	    float landmark_y = map_landmarks.landmark_list[j].y_f;
	    int landmark_id = map_landmarks.landmark_list[j].id_i;

	    double landmark_distance = dist(particle_x,particle_y,landmark_x,landmark_y);
	    if (landmark_distance <= sensor_range) landmarks_in_range.push_back({landmark_id, landmark_x, landmark_y});
          }  	

	  // Coordinate system transformation (Vehicle to Map's coordinate system)
	  vector<LandmarkObs> transformed_observations;
	  for (int k = 0; k < observations.size(); k++){
	    double x = observations[k].x * cos(particle_theta) - observations[k].y * sin(particle_theta) + particle_x;
	    double y = observations[k].x * sin(particle_theta) + observations[k].y * cos(particle_theta) + particle_y;
	    transformed_observations.push_back({ observations[k].id, x, y }); 
	  }

	  // Data association
	  dataAssociation(landmarks_in_range, transformed_observations);

	  // Particle Weights
	  particles[i].weight = 1.0;
	  for (int n = 0; n < transformed_observations.size(); n++){
	    
	    double predicted_x;
	    double predicted_y;

	    double observed_x = transformed_observations[n].x;
	    double observed_y = transformed_observations[n].y;
	    int associated_prediction = transformed_observations[n].id;

	    // Get ssociated predictions parameters  	
	    for (int m = 0; m < landmarks_in_range.size(); m++) {
              if (landmarks_in_range[m].id == associated_prediction) {
                predicted_x = landmarks_in_range[m].x;
                predicted_y = landmarks_in_range[m].y;
              }
            }

	    // Particle weights
	    particles[i].weight *=  MK * exp(-0.5 * (pow(predicted_x-observed_x,2)/std_x2 + pow(predicted_y-observed_y,2)/std_y2) ); 
	  }
	} 
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	// Random generator 
	random_device rd; 
	mt19937 gen(rd());
	
	// Get particle weights
	vector<double> particle_weights;
	for (int i = 0; i < num_particles; i++) particle_weights.push_back(particles[i].weight);
	
	// Discrete distribution
	discrete_distribution<> distribution(particle_weights.begin(), particle_weights.end());
	
	// Resampling particles
	vector<Particle> resampled_particles;
	resampled_particles.resize(num_particles);
	for (int j = 0; j < num_particles; j++) {
		int k = distribution(gen);
		resampled_particles[j] = particles[k];
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{	
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
