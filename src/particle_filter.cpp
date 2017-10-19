/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Modified: Ioannis Tornazakis
 */

#include "particle_filter.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "helper_functions.h"

using namespace std;

// Define constants
#define STD_X 0
#define STD_Y 1
#define STD_THETA 2
#define NUM_PARTICLES 100
#define INIT_WEIGHT 1
#define E1 0.0001

// Globals
const bool DEBUG = true;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the filter's number of particles
  num_particles = NUM_PARTICLES;
  
  // Create a normal Gaussian distributions for x, y, and theta
  normal_distribution<double> dist_x(x, std[STD_X]);
  normal_distribution<double> dist_y(y, std[STD_Y]);
  normal_distribution<double> dist_theta(theta, std[STD_THETA]);
  
  // Initialize the particles
  for (int ix = 0; ix < NUM_PARTICLES; ++ix) {
    Particle p;
    
    // Initialize each of the particles
    p.id = ix;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = INIT_WEIGHT;
    
    // Add the particle to the filter's particles vector
    particles.push_back(p);
  } // End for
  
  // Update the initialization flag
  is_initialized = true;
  
  if (DEBUG) {
    cout << "> Initialized " << particles.size() << " particles" << endl;
  } // End if
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  const double dt = delta_t;
  const double c1 = (yaw_rate) ? (velocity / yaw_rate) : 0.0;
  const double arc = yaw_rate * dt;
  const double ds = velocity * dt;
  
  // Create a normal Gaussian distributions for noise in x, y, theta
  normal_distribution<double> dist_x(0.0, std_pos[STD_X]);
  normal_distribution<double> dist_y(0.0, std_pos[STD_Y]);
  normal_distribution<double> dist_theta(0.0, std_pos[STD_THETA]);
  
  // Predict the future pose of the particles
  for (int ix = 0; ix < NUM_PARTICLES; ++ix) {
    const Particle p = particles[ix];
    
    // Create the noise component for x, y, theta
    const double n_x = dist_x(gen);
    const double n_y = dist_y(gen);
    const double n_theta = dist_theta(gen);
    
    // Avoid division by zero
    if (yaw_rate < E1) {
      // In this case the particle is moving straight
      particles[ix].x += ds * cos(p.theta) + n_x;
      particles[ix].y += ds * sin(p.theta) + n_y;
      particles[ix].theta += ds + n_theta;
    } else {
      particles[ix].x += c1 * (sin(p.theta + arc) - sin(p.theta)) + n_x;
      particles[ix].y += c1 * (cos(p.theta) + cos(p.theta + arc)) + n_y;
      particles[ix].theta += arc + n_theta;
    } // End if else
  } // End for
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  double min_distance = numeric_limits<double>::max();
  
  // For each observed measurement
  for (int ix = 0; ix < observations.size(); ++ix) {
    // Reset the current distance for each itteration
    double current_distance = min_distance;
    
    // Scan through all the predicted measurements
    for (int iy = 0; iy < predicted.size(); ++iy) {
      // Get the coordinates of the observed measurement
      double x1 = observations[ix].x;
      double y1 = observations[ix].y;
      
      // Get the coordinates of the predicted measurement
      double x2 = predicted[iy].x;
      double y2 = predicted[iy].y;
      
      // Compare the distance of the observed and the predicted measurements
      current_distance = dist(x1, y1, x2, y2);
      
      // Set the observation id to the nearest predicted landmark id
      if (current_distance < min_distance) {
        // Set the minimum distance to the new minimum
        min_distance = current_distance;
        
        // Set the observations id to the closest prediction yet
        observations[ix].id = predicted[iy].id;
      } // End if
    } // End inner for
  } // End outer for
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // For all the particles
  for (int ix = 0; ix < particles.size(); ++ix) {
    Particle p = particles[ix];
    
    // Local vectors
    vector<LandmarkObs> observations_map;
    
    // For each observation transform the coordinates from vehicle to map
    for (int iy = 0; iy < observations.size(); ++iy) {
      LandmarkObs obs = observations[iy];
     
      // Treansform coordinates using formula 3.33 from
      // http://planning.cs.uiuc.edu/node99.html
      LandmarkObs observation_map;
      observation_map.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
      observation_map.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
      observations_map.push_back(observation_map);
      
    } // End observations inner for
    
    // Locate the landmarks within the sensor range
    
    // Associate landmark in range (id) to landmark observations
    
    // Update the weights of each particle using a mult-variate Gaussian
    // distribution. You can read more about this distribution here:
    // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
      
  } // End outer for
  
  // Particles already in map coordinates
  
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to
  // their weight.
  
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle,
   std::vector<int> associations, std::vector<double> sense_x,
   std::vector<double> sense_y) {
  // Particle: the particle to assign each listed association, and
  // association's (x,y) world coordinates mapping to associations:
	// The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	// Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
