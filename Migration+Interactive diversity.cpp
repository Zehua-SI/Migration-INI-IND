#include <iostream>     // Standard input-output library
#include <fstream>      // File input-output library
#include <cstdlib>      // Standard library functions (e.g., rand(), srand())
#include <cstdio>       // Standard C I/O library
#include <cmath>        // Math functions (e.g., exp(), sqrt())
#include <ctime>        // Time library (used for random seed initialization)
#include <vector>       // Dynamic array container
#include <algorithm>    // Algorithm library (e.g., random_shuffle())

using namespace std;

// Define constants
#define L 100                // Grid size (length of one side)
#define N (L * L)            // Total number of nodes in the grid
#define neig_num 4           // Each node has 4 neighbors (top, bottom, left, right)

// Define global variables
vector<int> occupied_sites;  // List of occupied sites
vector<int> empty_sites;     // List of empty sites
int neighbors[N][neig_num];  // Neighbor indices for each node
double strategy[N];          // Strategy value of each player (0 to 1, representing cooperation probability)
bool is_movable[N];          // Whether a player is movable or not
bool has_moved[N];           // Whether a player has moved in the current round
double payoff[N];            // Stores each player's accumulated payoff

// Game parameters
double K = 0.1;              // Noise parameter in the Fermi update function
double r;                    // Public goods game multiplication factor
double c = 1.0;              // Cost of cooperation
double f = 3.0;                // Cost of movement

// Parameters for Mersenne Twister random number generator
#define NN 624
#define MM 397
#define MATRIX_A 0x9908b0df
#define UPPER_MASK 0x80000000
#define LOWER_MASK 0x7fffffff
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

// Buffer for Mersenne Twister RNG
static unsigned long mt_buffer[NN];
static int mti = NN + 1;  // Index for random number generation

// Initialize the random number generator with a seed
void sgenrand(unsigned long seed) {
    mt_buffer[0] = seed & 0xffffffff; // Set initial seed
    for (mti = 1; mti < NN; mti++)
        mt_buffer[mti] = (69069 * mt_buffer[mti - 1]) & 0xffffffff; // Linear congruential generator
}

// Generate a random number using Mersenne Twister
unsigned long genrand() {
    unsigned long y;
    static unsigned long mag01[2] = { 0x0, MATRIX_A };

    // If index exceeds the range, regenerate the numbers
    if (mti >= NN) {
        int kk;
        if (mti == NN + 1)
            sgenrand(4357); // Default seed initialization

        for (kk = 0; kk < NN - MM; kk++) {
            y = (mt_buffer[kk] & UPPER_MASK) | (mt_buffer[kk + 1] & LOWER_MASK);
            mt_buffer[kk] = mt_buffer[kk + MM] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (; kk < NN - 1; kk++) {
            y = (mt_buffer[kk] & UPPER_MASK) | (mt_buffer[kk + 1] & LOWER_MASK);
            mt_buffer[kk] = mt_buffer[kk + (MM - NN)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt_buffer[NN - 1] & UPPER_MASK) | (mt_buffer[0] & LOWER_MASK);
        mt_buffer[NN - 1] = mt_buffer[MM - 1] ^ (y >> 1) ^ mag01[y & 0x1];
        mti = 0; // Reset index
    }

    y = mt_buffer[mti++]; // Get the generated random number
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);

    return y; // Return the generated random number
}

// Generate a random floating-point number in the range [0,1)
double randf() { return ((double)genrand() * 2.3283064370807974e-10); }

// Generate a random integer in the range [0, LIM-1]
long randi(unsigned long LIM) { return ((unsigned long)genrand() % LIM); }

// Initialize neighbors for each node in the grid (with periodic boundary conditions)
void find_neig() {
    for (int i = 0; i < N; i++) {
        neighbors[i][0] = i - L; // Top neighbor
        neighbors[i][1] = i + L; // Bottom neighbor
        neighbors[i][2] = i - 1; // Left neighbor
        neighbors[i][3] = i + 1; // Right neighbor

        // Handle periodic boundary conditions
        if (i < L) neighbors[i][0] = i + L * (L - 1); // If at the top, wrap to bottom
        if (i >= L * (L - 1)) neighbors[i][1] = i - L * (L - 1); // If at the bottom, wrap to top
        if (i % L == 0) neighbors[i][2] = i + L - 1; // If at the leftmost column, wrap to the rightmost column
        if (i % L == L - 1) neighbors[i][3] = i - L + 1; // If at the rightmost column, wrap to the leftmost column
    }
}

// Global counters for tracking the number of players in different categories
int total_movable = 0;       // Count of movable players
int total_non_movable = 0;   // Count of non-movable players
int total_occupied = 0;      // Count of occupied nodes
int movable_moved_count = 0; // Number of movable players who actually moved in the current round

// Initialize the game with a given density (rho)
void init_game(double rho) {
    find_neig(); // Initialize neighbor relationships

    occupied_sites.clear(); // Clear the list of occupied sites
    empty_sites.clear();    // Clear the list of empty sites

    total_movable = 0;
    total_non_movable = 0;
    total_occupied = 0;

    // Loop through all nodes to determine whether they are occupied or empty
    for (int i = 0; i < N; i++) {
        if (randf() < rho) { // A site is occupied with probability `rho`
            is_movable[i] = (randf() < 0.5); // 50% probability of being a movable player
            occupied_sites.push_back(i);     // Add to occupied sites list
            if (is_movable[i]) {
                total_movable++;
            } else {
                total_non_movable++;
            }
            total_occupied++;
        } else {
            is_movable[i] = false;  // Empty nodes are not movable
            strategy[i] = -1.0;     // Set empty node strategy value to -1
            empty_sites.push_back(i); // Add to empty sites list
        }
    }

    // Second pass: Determine each occupied player's initial strategy based on neighbors
    for (int i : occupied_sites) {
        int occupied_neighbors = 0;  // Count of occupied neighbors
        int coop_count = 0;          // Cooperation count

        // Loop through the player's four neighbors
        for (int j = 0; j < neig_num; j++) {
            int neig = neighbors[i][j];
            if (strategy[neig] != -1.0) { // If the neighbor is occupied
                occupied_neighbors++;
                if (randf() < 0.5) {  // 50% probability to cooperate
                    coop_count++;
                }
            }
        }

        // Compute initial strategy value
        if (occupied_neighbors > 0) {
            strategy[i] = ((double)coop_count) / occupied_neighbors; // Cooperation probability
        } else {
            strategy[i] = 0.5; // Default strategy is 0.5 if no neighbors exist
        }
    }
}

// Determine whether a player should move
bool can_move(int x) {
    if (!is_movable[x]) return false;  // Non-movable players cannot move

    int occupied_neighbors = 0;       // Count of non-empty neighbors
    double avg_strategy = 0.0;        // Average strategy value of non-empty neighbors

    // Check all four neighbors
    for (int i = 0; i < neig_num; i++) {
        int neig = neighbors[x][i];
        if (strategy[neig] != -1.0) {  // If the neighbor is occupied
            occupied_neighbors++;      // Increase count of non-empty neighbors
            avg_strategy += strategy[neig]; // Sum up neighbor strategy values
        }
    }

    // Compute the average strategy value of non-empty neighbors
    if (occupied_neighbors > 0) {
        avg_strategy /= occupied_neighbors;
    }

    // Move if all neighbors are empty OR if the average strategy value is too low
    return (occupied_neighbors == 0 || avg_strategy < 0.5);
}

// Move a player to a random empty site
void move_player(int x) {
    if (!empty_sites.empty()) {
        int index = randi(empty_sites.size());  // Randomly select an empty site
        int new_pos = empty_sites[index];       // Get the position of the empty site

        // Transfer strategy and mobility status to the new position
        strategy[new_pos] = strategy[x];
        is_movable[new_pos] = is_movable[x];

        // Mark the old position as empty
        strategy[x] = -1.0;
        is_movable[x] = false;

        // Update the empty sites list
        empty_sites[index] = x;  // Replace the moved position with the old position

        // Update the occupied sites list
        auto it = find(occupied_sites.begin(), occupied_sites.end(), x);
        if (it != occupied_sites.end()) {
            *it = new_pos;  // Replace the old position with the new one
        }

        // Mark the player as moved in this round
        has_moved[new_pos] = true;

        // Increase count of players who actually moved
        movable_moved_count++;
    }
}

// Calculate the accumulated payoff for a player
double cal_payoff(int x) {
    double total_payoff = 0.0;  // Initialize total payoff

    // First, consider the public goods game centered on the focal player
    vector<int> group_members;
    group_members.push_back(x);  // Add the focal player to the group

    // Add non-empty neighbors to the group
    for (int i = 0; i < neig_num; i++) {
        int neig = neighbors[x][i];
        if (strategy[neig] != -1.0) {  // Check if the neighbor is occupied
            group_members.push_back(neig);
        }
    }

    // If the group only consists of the focal player, set payoff to zero
    if (group_members.size() > 1) {
        int cooperators = 0;

        // Determine cooperation decisions based on probabilities
        bool x_cooperates = (randf() < strategy[x]);
        if (x_cooperates) cooperators++;

        for (int player : group_members) {
            if (player == x) continue;
            if (randf() < strategy[player]) cooperators++;
        }

        // Compute benefits and cost
        double total_benefit = r * cooperators;
        double payoff_i = total_benefit / group_members.size();
        if (x_cooperates) {  // If the focal player cooperates, subtract cost
            payoff_i -= c;
        }
        total_payoff += payoff_i;
    }

    // Now, consider the public goods games centered on the player's neighbors
    for (int i = 0; i < neig_num; i++) {
        int focal = neighbors[x][i];  // Get the neighbor as focal player
        if (strategy[focal] == -1.0) continue;  // Skip empty neighbors

        group_members.clear();
        group_members.push_back(focal);  // Add neighbor as group center

        // Add non-empty neighbors of the neighbor to the group
        for (int j = 0; j < neig_num; j++) {
            int neig = neighbors[focal][j];
            if (strategy[neig] != -1.0) {
                group_members.push_back(neig);
            }
        }

        // Compute payoff for the focal player's group
        if (group_members.size() > 1) {
            int cooperators = 0;
            bool x_cooperates_in_focal_game = (randf() < strategy[x]);
            if (x_cooperates_in_focal_game) cooperators++;

            for (int player : group_members) {
                if (player == x) continue;
                if (randf() < strategy[player]) cooperators++;
            }

            double total_benefit = r * cooperators;
            double payoff_i = total_benefit / group_members.size();
            if (x_cooperates_in_focal_game) {
                payoff_i -= c;
            }
            total_payoff += payoff_i;
        }
    }

    // If the player moved in this round, subtract movement cost
    if (has_moved[x]) {
        total_payoff -= f;
    }

    return total_payoff;
}

// Allow players to learn strategies from neighbors
void learn_strategy(int x) {
    vector<int> occupied_neighbors; // Store occupied neighbors

    // Collect all occupied neighbors
    for (int i = 0; i < neig_num; i++) {
        int neig = neighbors[x][i];
        if (strategy[neig] != -1.0) {
            occupied_neighbors.push_back(neig);
        }
    }

    if (occupied_neighbors.empty()) return; // If no neighbors, no learning

    // Randomly select a neighbor
    int neig = occupied_neighbors[randi(occupied_neighbors.size())];

    double x_payoff = payoff[x];        // Own payoff
    double neig_payoff = payoff[neig];  // Neighbor's payoff

    // Compute update probability using Fermi function
    double prob = 1.0 / (1.0 + exp((x_payoff - neig_payoff) / K));

    // With probability `prob`, adopt the neighbor's strategy
    if (randf() < prob) {
        strategy[x] = strategy[neig];
    }
}

// Execute one round of the game
void round_game() {
    // Reset movement status at the beginning of each round
    for (int i = 0; i < N; i++) {
        has_moved[i] = false;
    }
    movable_moved_count = 0;

    vector<int> players_to_move; // List of players who need to move

    // Collect players who need to move
    for (int i : occupied_sites) {
        if (can_move(i)) {
            players_to_move.push_back(i);
        }
    }

    // Shuffle movement order
    random_shuffle(players_to_move.begin(), players_to_move.end());

    // Move players
    for (int i : players_to_move) {
        move_player(i);
    }

    // Calculate payoffs for all players
    for (int i : occupied_sites) {
        if (strategy[i] >= 0.0 && strategy[i] <= 1.0) {
            payoff[i] = cal_payoff(i);
        }
    }

    // Strategy update phase
    vector<int> current_occupied = occupied_sites; // Copy occupied sites list

    for (int i : current_occupied) {
        if (strategy[i] != -1.0) {
            learn_strategy(i);
        }
    }
}

// Arrays for output: average cooperation fractions and payoffs for different player categories
double data_out[3];       // Stores average cooperation fractions
double data_payoff[3];    // Stores average payoffs

// Compute statistics for cooperation fractions and payoffs
void cal_data() {
    double total_movable_strategy = 0.0;    // Sum of strategy values for movable players
    int movable_count = 0;                  // Count of movable players

    double total_non_movable_strategy = 0.0; // Sum of strategy values for non-movable players
    int non_movable_count = 0;               // Count of non-movable players

    double total_occupied_strategy = 0.0;   // Sum of strategy values for all occupied nodes
    int occupied_count = 0;                 // Count of occupied nodes

    double total_movable_payoff = 0.0;      // Sum of payoffs for movable players
    double total_non_movable_payoff = 0.0;  // Sum of payoffs for non-movable players
    double total_occupied_payoff = 0.0;     // Sum of payoffs for all occupied nodes

    // Iterate through all occupied sites
    for (int i : occupied_sites) {
        if (strategy[i] >= 0.0 && strategy[i] <= 1.0) {
            total_occupied_strategy += strategy[i]; // Sum up strategy values
            total_occupied_payoff += payoff[i];     // Sum up payoffs
            occupied_count++;

            if (is_movable[i]) {  // Movable players
                total_movable_strategy += strategy[i];
                total_movable_payoff += payoff[i];
                movable_count++;
            } else {  // Non-movable players
                total_non_movable_strategy += strategy[i];
                total_non_movable_payoff += payoff[i];
                non_movable_count++;
            }
        }
    }

    // Compute average cooperation fractions
    data_out[0] = (movable_count > 0) ? (total_movable_strategy / movable_count) : 0.0;
    data_out[1] = (non_movable_count > 0) ? (total_non_movable_strategy / non_movable_count) : 0.0;
    data_out[2] = (occupied_count > 0) ? (total_occupied_strategy / occupied_count) : 0.0;

    // Compute average payoffs
    data_payoff[0] = (movable_count > 0) ? (total_movable_payoff / movable_count) : 0.0;
    data_payoff[1] = (non_movable_count > 0) ? (total_non_movable_payoff / non_movable_count) : 0.0;
    data_payoff[2] = (occupied_count > 0) ? (total_occupied_payoff / occupied_count) : 0.0;
}

// Compute standard deviation
void calculate_standard_deviation(const vector<double> &experiment_results, double mean, double &std_dev) {
    double variance = 0.0;
    for (double result : experiment_results) {
        variance += (result - mean) * (result - mean);
    }
    variance /= experiment_results.size();  // Compute variance
    std_dev = sqrt(variance);               // Compute standard deviation
}

int main(void) {
    int Round = 1000000;          // Total rounds per experiment
    int Experiments = 100;        // Number of independent experiments

    sgenrand(time(0));          // Initialize random seed

    printf("***** Start Simulation *****\n");

    // Open CSV file for storing results
    FILE *Fc = fopen("Migration+Interactive diversity.csv", "w");
    if (Fc == NULL) {
        perror("Failed to open file");
        return EXIT_FAILURE;
    }

    // Write CSV file header
    fprintf(Fc, "rho,r,Final_Fc_m,Final_Fc_nm,Final_Fc,Std_Fc_m,Std_Fc_nm,Std_Fc,"
                "Final_Payoff_m,Final_Payoff_nm,Final_Payoff,Std_Payoff_m,Std_Payoff_nm,Std_Payoff\n");

    // Loop over density (rho) from 0.10 to 1.00 in increments of 0.01
    for (double rho = 0.10; rho <= 1.00; rho += 0.01) {
        // Loop over public goods multiplication factor (r) from 3.0 to 5.0 in increments of 0.01
        for (r = 3.0; r <= 5.00; r += 0.01) {
            double total_data[3] = { 0.0, 0.0, 0.0 }; // Stores sum of cooperation fractions
            double total_payoff_data[3] = { 0.0, 0.0, 0.0 }; // Stores sum of payoffs
            vector<double> exp_results[3];  // Store cooperation fractions for all experiments
            vector<double> exp_payoff_results[3]; // Store payoffs for all experiments

            // Run multiple independent experiments
            for (int exp = 0; exp < Experiments; exp++) {
                init_game(rho); // Initialize the game state

                double data_sum[3] = { 0.0, 0.0, 0.0 }; // Sum of cooperation fractions over last 5000 steps
                double data_sum_payoff[3] = { 0.0, 0.0, 0.0 }; // Sum of payoffs over last 5000 steps

                // Run the simulation for the specified number of rounds
                for (int i = 0; i < Round; i++) {
                    round_game();   // Execute one round of the game
                    cal_data();     // Compute cooperation and payoff statistics

                    // Collect data only for the last 5000 rounds
                    if (i >= Round - 5000) {
                        for (int j = 0; j < 3; j++) {
                            data_sum[j] += data_out[j];
                            data_sum_payoff[j] += data_payoff[j];
                        }
                    }
                }

                // Compute average cooperation fraction and average payoff over last 5000 rounds
                for (int j = 0; j < 3; j++) {
                    double avg_strategy = data_sum[j] / 5000;
                    total_data[j] += avg_strategy;
                    exp_results[j].push_back(avg_strategy);

                    double avg_payoff = data_sum_payoff[j] / 5000;
                    total_payoff_data[j] += avg_payoff;
                    exp_payoff_results[j].push_back(avg_payoff);
                }
            }

            // Compute mean and standard deviation for each category
            double mean[3], std_dev[3];
            double mean_payoff[3], std_dev_payoff[3];
            for (int j = 0; j < 3; j++) {
                mean[j] = total_data[j] / Experiments;
                calculate_standard_deviation(exp_results[j], mean[j], std_dev[j]);

                mean_payoff[j] = total_payoff_data[j] / Experiments;
                calculate_standard_deviation(exp_payoff_results[j], mean_payoff[j], std_dev_payoff[j]);
            }

            // Write results to the CSV file
            fprintf(Fc, "%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                        "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    rho, r, mean[0], mean[1], mean[2], std_dev[0], std_dev[1], std_dev[2],
                    mean_payoff[0], mean_payoff[1], mean_payoff[2], std_dev_payoff[0], std_dev_payoff[1], std_dev_payoff[2]);

            // Print summary to the console
            printf("rho = %.2f, r = %.2f: Final_Fc_m = %.4f, Final_Fc_nm = %.4f, Final_Fc = %.4f, "
                   "Final_Payoff_m = %.4f, Final_Payoff_nm = %.4f, Final_Payoff = %.4f\n",
                   rho, r, mean[0], mean[1], mean[2], mean_payoff[0], mean_payoff[1], mean_payoff[2]);
        }
    }

    fclose(Fc); // Close the output file
    printf("***** Simulation Complete *****\n");
    return 0;
}
