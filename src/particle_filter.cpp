/**
 * particle_filter.cpp
 *
 * Created on: October 7, 2017
 * Author: Ioannis Tornazakis
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

// Globals
const int STD_X = 0;
const int STD_Y = 1;
const int STD_THETA = 2;
const int NUM_PARTICLES = 100;
const double INIT_WEIGHT = 1.0;
const double E1 = 0.0001;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the filter's number of particles
  num_particles = NUM_PARTICLES;
  
  // Create normal Gaussian distributions for x, y, and theta
  normal_distribution<double> dist_x(x, std[STD_X]);
  normal_distribution<double> dist_y(y, std[STD_Y]);
  normal_distribution<double> dist_theta(theta, std[STD_THETA]);
  
  // Initialize the particles with the default attributes and add them to
  // the list of particles
  for (int ix = 0; ix < NUM_PARTICLES; ++ix) {
    Particle p;
    p.id = ix;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = INIT_WEIGHT;
    particles.push_back(p);
  } // End for
  
  // Mark the filter as initialized
  is_initialized = true;
}

void ParticleFilter::prediction(double dt, double std_pos[], double v,
                                double yaw_rate) {
  // Create a normal Gaussian distributions for noise in x, y, theta
  normal_distribution<double> dist_x(0.0, std_pos[STD_X]);
  normal_distribution<double> dist_y(0.0, std_pos[STD_Y]);
  normal_distribution<double> dist_theta(0.0, std_pos[STD_THETA]);
  
  // Predict the future pose of the particles
  for (int ix = 0; ix < particles.size(); ++ix) {
    // Create a pointer to the particle element to access its attributes
    Particle * p = &particles[ix];
    const double theta = (*p).theta;
    
    // Create the noise components for x, y, theta
    const double n_x = dist_x(gen);
    const double n_y = dist_y(gen);
    const double n_theta = dist_theta(gen);
    
    //--------------------------------------------------------------------------
    // Equations with (θ' !=  0):
    //--------------------------------------------------------------------------
    //      υ
    // x = --- * (sin(θ + θ' * dt) - sin(θ)) + νx                   (1)
    //      θ'
    //
    //      υ
    // y = --- * (cos(θ) - cos(θ + θ' * dt)) + νy                   (2)
    //      θ'
    //
    //--------------------------------------------------------------------------
    // Equations for straight motion (θ' = 0):
    //--------------------------------------------------------------------------
    //
    // x = υ * cos(θ) * dt + νx                                     (3)
    //
    // y = υ * sin(θ) * dt + νy                                     (4)
    //
    //--------------------------------------------------------------------------
    
    // Avoid division by zero in equations (1) and (2)
    if (fabs(yaw_rate) < E1) {
      (*p).x += v * dt * cos(theta) + n_x; // (3)
      (*p).y += v * dt * sin(theta) + n_y; // (4)
      (*p).theta += n_theta;
    } else {
      (*p).x += (v / yaw_rate) * (sin(theta + yaw_rate * dt) - sin(theta))
                + n_x; // (1)
      (*p).y += (v / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * dt))
                + n_y; // (2)
      (*p).theta += yaw_rate * dt + n_theta;
    } // End if/else - calculations
  } // End for - particles
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // Scan through all observations and predicted landmarks to find the
  // closest match
  for (int ix = 0; ix < observations.size(); ++ix) {
    LandmarkObs * obs = &observations[ix];
    
    // Reset the minimum distance for each itteration to a large number
    double min_distance = numeric_limits<double>::max();
    
    // Compare the distance of the observed and the predicted measurements.
    // When finding a mimimum, update the observation id to the corresponding
    // prediction
    for (int iy = 0; iy < predicted.size(); ++iy) {
      const LandmarkObs * pred = &predicted[iy];
      double current_distance = dist((*obs).x, (*obs).y, (*pred).x, (*pred).y);
      
      if (min_distance > current_distance) {
        min_distance = current_distance;
        (*obs).id = (*pred).id;
      } // End if - min_distance
      
    } // End for - predicted
  } // End for - observations
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // For all the particles
  for (int ix = 0; ix < particles.size(); ++ix) {
    // Create a pointer to access each particle during the loop
    Particle * p = &particles[ix];
    
    // Make sure the weight is set to 1 for the multiplication rule during
    // the update phase to work as expected
    (*p).weight = INIT_WEIGHT;
    
    //*************************************************************************
    // 1. Observation coordinates transformation from vehicle to map
    //*************************************************************************
    
    vector<LandmarkObs> observations_map;
    for (int iy = 0; iy < observations.size(); ++iy) {
      const LandmarkObs obs = observations[iy];
      LandmarkObs observation_map;
      
      //------------------------------------------------------------------------
      // Coordinates transformation from vehicle to map:
      //------------------------------------------------------------------------
      //
      // x_map = x_obs * cos(θ) - y_obs * sin(θ)                      (5)
      //
      // y_map = x_obs * sin(θ) + y_obs * cos(θ)                      (6)
      //
      //------------------------------------------------------------------------
      
      double x = (*p).x;
      double y = (*p).y;
      double theta = (*p).theta;
      observation_map.x = x + obs.x * cos(theta) - obs.y * sin(theta); // (5)
      observation_map.y = y + obs.x * sin(theta) + obs.y * cos(theta); // (6)
      observations_map.push_back(observation_map);
    } // End inner for - observations
    
    //*************************************************************************
    // 2. Locate landmarks within sensor range
    //*************************************************************************
    
    vector<LandmarkObs> landmarks_within_range;
    for (int iz = 0; iz < map_landmarks.landmark_list.size(); ++iz) {
      // Map landmark to get information
      Map::single_landmark_s map_landmark = map_landmarks.landmark_list[iz];
      
      // Observation landmark to append to the landmarks within sensor range
      LandmarkObs obs_landmark;
      
      // Get the coordinates of the particle
      double x1 = (*p).x;
      double y1 = (*p).y;
      
      // Get the coordinates of the landmark
      int id = map_landmark.id_i;
      double x2 = map_landmark.x_f;
      double y2 = map_landmark.y_f;
      
      // If the landmark is within sensor range, keep it in the list
      if (dist(x1, y1, x2, y2) < sensor_range) {
        LandmarkObs obs_landmark = {id, x2, y2};
        landmarks_within_range.push_back(obs_landmark);
      } // End if - distance
    } // End inner for - map landmarks
    
    //*************************************************************************
    // 3. Associate landmark in range (id) to landmark observations
    //*************************************************************************
    
    dataAssociation(landmarks_within_range, observations_map);
    
    //*************************************************************************
    // 4. Update the weights
    //*************************************************************************
    
    //--------------------------------------------------------------------------
    // Mult-variate Gaussian distribution equation:
    //--------------------------------------------------------------------------
    //
    //                1              (x - μχ)^2   (y - μy)^2
    //  P(x,y) = ------------ * e^-( ---------- + ---------- )          (7)
    //           2π * σx * σy         2 * σx^2     2 * σy^2
    //
    //--------------------------------------------------------------------------
    
    for (int ik = 0; ik < observations_map.size(); ++ik) {
      // (x, y) are the coordinates of the observations transformed to map
      // coordinates
      const LandmarkObs * obs = &observations_map[ik];
      double x = (*obs).x;
      double y = (*obs).y;
      
      // Locate the nearest landmark to the current observation
      LandmarkObs lm;
      for (int iv = 0; iv < landmarks_within_range.size(); iv++) {
        if (landmarks_within_range[iv].id == (*obs).id) {
           lm = landmarks_within_range[iv];
        } // End if
      } // End for
      
      // (μx, μy) are the coordinates of the nearest landmark (already in map
      // coordinates) to the observation
      double mu_x = lm.x;
      double mu_y = lm.y;
      
      // Short for the standard deviations
      double sx = std_landmark[STD_X];
      double sy = std_landmark[STD_Y];
      
      // Implement the multivariate Gaussian equation (7)
      double c1 = (1.0 / (2.0 * M_PI * sx * sy));
      double x_fract = pow(x - mu_x, 2.0) / (2.0 * pow(sx, 2));
      double y_fract = pow(y - mu_y, 2.0) / (2.0 * pow(sy, 2));
      double w = c1 * exp(-(x_fract + y_fract));
      
      // Multiply each weight to get the particle's total weight
      (*p).weight *= w;
    } // End inner for - update weights
    
    // Append to the all inclusive weights list
    weights.push_back((*p).weight);
  } // End outer for
}

void ParticleFilter::resample() {
  // Vector to hold the particles after resampling
  vector<Particle> resampled_particles;
  
  // Create a random distribution with bias towards the weights
  random_device rd;
  mt19937 gen(rd());
  discrete_distribution<> weight_dist(weights.begin(), weights.end());
  
  // For each particle position select a new one according to the distribution
  for (int ix = 0; ix < NUM_PARTICLES; ++ix) {
      Particle p;
      int iy = weight_dist(gen);
      resampled_particles.push_back(particles[iy]);
  }
  
  // Update the particles and reset the weights
  particles = resampled_particles;
  weights.clear();
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
