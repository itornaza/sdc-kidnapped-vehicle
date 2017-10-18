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

using namespace std;

// Define constants
#define STD_X 0
#define STD_Y 1
#define STD_THETA 2
#define NUM_PARTICLES 100
#define INIT_WEIGHT 1

// Globals
const bool DEBUG = true;

/**
 * init Initializes all particles to the first position (based on estimates of
 * x, y, theta and their uncertainties from GPS). Sets all weights to 1 and
 * adds random Gaussian noise to each particle
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Create a random engine to pick samples from
  default_random_engine gen;
  
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
  }
  
  // Update the initialization flag
  is_initialized = true;
  
  if (DEBUG) {
    cout << "> Initialized " << particles.size() << " particles" << endl;
  }
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
  
	// NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
	// http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	// http://www.cplusplus.com/reference/random/default_random_engine/

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the observed measurement to this particular
  // landmark.
	
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to implement this method and use it as a helper
  // during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read more about this distribution here:
  // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	
  // NOTE: The observations are given in the VEHICLE'S coordinate system.
  // Your particles are located according to the MAP'S coordinate system. You
  // will need to transform between the two systems. Keep in mind that this
  // transformation requires both rotation AND translation (but no scaling).
	// The following is a good resource for the theory:
	// https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	// and the following is a good resource for the actual equation to implement
  // (look at equation 3.33
	// http://planning.cs.uiuc.edu/node99.html
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
