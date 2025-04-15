#include <iostream>     // Include iostream for standard input/output
#include <fstream>      // Include fstream for file input/output
#include <cstdlib>      // Include cstdlib for functions such as rand(), srand(), etc.
#include <cstdio>       // Include cstdio for functions like printf, fopen, etc.
#include <cmath>        // Include cmath for math functions like exp(), sqrt(), etc.
#include <ctime>        // Include ctime for system time (e.g., initializing random seeds)
#include <vector>       // Include vector for dynamic array containers
#include <algorithm>    // Include algorithm for functions like random_shuffle()

using namespace std;    // Use the std namespace for convenience

//--------------------------- 1. Constants and Global Variables ---------------------------//
#define L 400                // Lattice length
#define N (L * L)            // Total number of nodes (for instance, 100*100=10000)
#define neig_num 4           // Number of neighbors per node (up, down, left, right)

// Global containers and arrays for storing node-related information
vector<int> occupied_sites;  // Stores indices of occupied nodes
vector<int> empty_sites;     // Stores indices of empty nodes
int neighbors[N][neig_num];  // For each node, stores the indices of its 4 neighbors (up, down, left, right)
double strategy[N];          // Each player's strategy: 1.0 for cooperator, 0.0 for defector, -1.0 for empty
bool is_movable[N];          // Indicates whether the player at each node is movable (true) or not (false)
bool has_moved[N];           // Indicates whether the player has moved in the current round
double payoff[N];            // Stores each player's accumulated payoff in the current round

// -------- Newly added or modified global variables -------------
double r_norm;         // Normalized cooperation factor for the public goods game (used in the loop)
double k_norm = 0.1;   // Fixed normalized noise parameter

// Used to record the group size when player x is the center of a public goods game (number of participants)
int local_group_size[N];

// Game parameters
double c = 1.0;         // Cost of cooperation
double f = 3.0;         // Cost of moving

// The following define parameters for the Mersenne Twister RNG
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

// Define the RNG buffer
static unsigned long mt_buffer[NN];
static int mti = NN + 1;  // The current RNG index; if > NN it means uninitialized

// Initialize the RNG with a given seed (using a linear congruential approach for initial filling)
void sgenrand(unsigned long seed) {
    mt_buffer[0] = seed & 0xffffffff;
    for (mti = 1; mti < NN; mti++)
        mt_buffer[mti] = (69069 * mt_buffer[mti - 1]) & 0xffffffff;
}

// Generate a random number using the Mersenne Twister algorithm
unsigned long genrand() {
    unsigned long y;
    static unsigned long mag01[2] = { 0x0, MATRIX_A };

    // If the current index exceeds NN, regenerate the random sequence
    if (mti >= NN) {
        int kk;
        if (mti == NN + 1)
            sgenrand(4357); // If not yet initialized, use default seed 4357

        // Process the first NN-MM numbers
        for (kk = 0; kk < NN - MM; kk++) {
            y = (mt_buffer[kk] & UPPER_MASK) | (mt_buffer[kk + 1] & LOWER_MASK);
            mt_buffer[kk] = mt_buffer[kk + MM] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        // Process the remaining numbers
        for (; kk < NN - 1; kk++) {
            y = (mt_buffer[kk] & UPPER_MASK) | (mt_buffer[kk + 1] & LOWER_MASK);
            mt_buffer[kk] = mt_buffer[kk + (MM - NN)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        // Special treatment for the last number
        y = (mt_buffer[NN - 1] & UPPER_MASK) | (mt_buffer[0] & LOWER_MASK);
        mt_buffer[NN - 1] = mt_buffer[MM - 1] ^ (y >> 1) ^ mag01[y & 0x1];
        mti = 0;
    }

    // Extract a random number
    y = mt_buffer[mti++];
    // Perform tempering transformations
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);

    return y;
}

// Generate a random floating-point number in [0,1)
double randf() {
    // 2.3283064370807974e-10 is 1/2^32, normalizing the integer to [0,1)
    return ((double)genrand() * 2.3283064370807974e-10);
}

// Generate a random integer in [0, LIM-1]
long randi(unsigned long LIM) {
    return ((unsigned long)genrand() % LIM);
}

// Initialize neighbors of each node (using periodic boundary conditions, forming a 2D lattice)
void find_neig() {
    for (int i = 0; i < N; i++) {
        // Set up, down, left, and right neighbors (not yet considering boundaries)
        neighbors[i][0] = i - L; // Up
        neighbors[i][1] = i + L; // Down
        neighbors[i][2] = i - 1; // Left
        neighbors[i][3] = i + 1; // Right

        // Handle periodic boundary conditions
        // For the first row, the top neighbor comes from the last row
        if (i < L) neighbors[i][0] = i + L * (L - 1);
        // For the last row, the bottom neighbor comes from the first row
        if (i >= L * (L - 1)) neighbors[i][1] = i - L * (L - 1);
        // For the first node in each row, the left neighbor is the last node in that row
        if (i % L == 0) neighbors[i][2] = i + L - 1;
        // For the last node in each row, the right neighbor is the first node in that row
        if (i % L == L - 1) neighbors[i][3] = i - L + 1;
    }
}

// Define global counters to track the number of different types of players
int total_movable = 0;       // Total number of movable players
int total_non_movable = 0;   // Total number of non-movable players
int total_occupied = 0;      // Total number of occupied nodes
int movable_moved_count = 0; // The number of movable players who actually moved in the current round

// Initialize the game based on the given density rho
void init_game(double rho) {
    find_neig(); // First initialize neighbor relationships

    occupied_sites.clear(); // Clear the list of occupied nodes
    empty_sites.clear();    // Clear the list of empty nodes

    // Reset counters
    total_movable = 0;
    total_non_movable = 0;
    total_occupied = 0;

    // Iterate over all nodes; decide if a node is occupied based on density rho
    for (int i = 0; i < N; i++) {
        if (randf() < rho) {  // If a random number < rho, the node is occupied
            // Randomly decide whether the player is movable (50% chance)
            is_movable[i] = (randf() < 0.5);
            occupied_sites.push_back(i);
            if (is_movable[i]) {
                total_movable++;
            } else {
                total_non_movable++;
            }
            total_occupied++;

            // Randomly set player strategy: 50% chance cooperator (1.0), 50% chance defector (0.0)
            strategy[i] = (randf() < 0.5) ? 1.0 : 0.0;
        } else {
            // If the node is not occupied, mark it as empty
            is_movable[i] = false;
            strategy[i] = -1.0;  // -1.0 indicates an empty node
            empty_sites.push_back(i);
        }
    }
}

// Check if player x meets the condition to move
bool can_move(int x) {
    if (!is_movable[x]) return false;  // If the player is not movable, they cannot move

    int occupied_neighbors = 0;    // Count how many neighbors are occupied
    int cooperators_neighbors = 0; // Count how many neighbors are cooperators

    // Check the four neighbors of x
    for (int i = 0; i < neig_num; i++) {
        int neig = neighbors[x][i];
        if (strategy[neig] != -1.0) {  // If the neighbor node is occupied
            occupied_neighbors++;
            if (strategy[neig] == 1.0) { // If the neighbor is a cooperator
                cooperators_neighbors++;
            }
        }
    }

    // Calculate the proportion of cooperators among neighbors
    double coop_ratio = (occupied_neighbors > 0) ?
                        ((double)cooperators_neighbors / occupied_neighbors) : 0.0;

    // If all neighbors are empty or the cooperator ratio is below 0.5, return that x should move
    return (occupied_neighbors == 0 || coop_ratio < 0.5);
}

// Move player x to a randomly selected empty node
void move_player(int x) {
    // Make sure there is at least one empty node
    if (!empty_sites.empty()) {
        int index = randi(empty_sites.size());  // Randomly pick an index in empty_sites
        int new_pos = empty_sites[index];       // Get the new position

        // Copy the strategy and mobility from x to new_pos
        strategy[new_pos] = strategy[x];
        is_movable[new_pos] = is_movable[x];

        // Mark the old position x as empty (strategy = -1.0, not movable)
        strategy[x] = -1.0;
        is_movable[x] = false;

        // Update the empty node list: replace that empty slot with x
        empty_sites[index] = x;

        // Update the occupied node list: find x and replace it with new_pos
        auto it = find(occupied_sites.begin(), occupied_sites.end(), x);
        if (it != occupied_sites.end()) {
            *it = new_pos;
        }

        // Mark that the player at new_pos has moved this round
        has_moved[new_pos] = true;

        // Increment the counter (tracking how many players actually moved this round)
        movable_moved_count++;
    }
}

// Calculate the accumulated payoff for player x in the current round
double cal_payoff(int x) {
    double total_payoff_x = 0.0;

    // --------------- Public goods game centered on x ---------------
    vector<int> group_members;
    group_members.push_back(x); // Add x to the game group

    // Add x's non-empty neighbors to the group
    for (int i = 0; i < neig_num; i++) {
        int neig = neighbors[x][i];
        if (strategy[neig] != -1.0) {
            group_members.push_back(neig);
        }
    }

    // If the group has more than one member, compute the public goods payoff
    if (group_members.size() > 1) {
        int cooperators = 0; // Count how many cooperators in the group
        for (int player : group_members) {
            if (strategy[player] == 1.0) {
                cooperators++;
            }
        }

        // Compute the actual cooperation factor: r_actual = r_norm * G
        int G = (int)group_members.size();
        double r_actual = r_norm * G;
        // Total payoff is r_actual times the number of cooperators
        double total_benefit = r_actual * cooperators;

        // Each participant gets an equal share
        double payoff_i = total_benefit / G;
        // If x is a cooperator, subtract the cooperation cost c
        if (strategy[x] == 1.0) {
            payoff_i -= c;
        }
        total_payoff_x += payoff_i;

        // Record the group size for x, for use in noise normalization during strategy updates
        local_group_size[x] = G;
    } else {
        // If only x is in the group, no public goods payoff, but still record group size = 1
        local_group_size[x] = 1;
    }

    // --------------- Public goods games centered on x's neighbors ---------------
    // x also participates in each neighbor's game
    for (int i = 0; i < neig_num; i++) {
        int focal = neighbors[x][i];
        if (strategy[focal] == -1.0) continue;  // If focal is empty, skip

        group_members.clear();
        group_members.push_back(focal); // Center on focal

        // Add focal's non-empty neighbors
        for (int j = 0; j < neig_num; j++) {
            int neig2 = neighbors[focal][j];
            if (strategy[neig2] != -1.0) {
                group_members.push_back(neig2);
            }
        }

        if (group_members.size() > 1) {
            int cooperators = 0;
            for (int player : group_members) {
                if (strategy[player] == 1.0) {
                    cooperators++;
                }
            }

            // Compute payoff with the normalized cooperation factor
            int G_focal = (int)group_members.size();
            double r_actual_focal = r_norm * G_focal;
            double total_benefit_focal = r_actual_focal * cooperators;

            double payoff_i = total_benefit_focal / G_focal;
            if (strategy[x] == 1.0) {
                payoff_i -= c;
            }
            total_payoff_x += payoff_i;
        }
    }

    // If player x moved this round, subtract the moving cost f
    if (has_moved[x]) {
        total_payoff_x -= f;
    }

    return total_payoff_x;
}

// Players update their strategy by learning from a neighbor's payoff
void learn_strategy(int x) {
    vector<int> occupied_neighbors;

    // Collect all non-empty neighbors of x
    for (int i = 0; i < neig_num; i++) {
        int neig = neighbors[x][i];
        if (strategy[neig] != -1.0) {
            occupied_neighbors.push_back(neig);
        }
    }

    // If x has no neighbors, return
    if (occupied_neighbors.empty()) return;

    // Randomly choose one neighbor as the learning target
    int neig = occupied_neighbors[randi(occupied_neighbors.size())];

    double x_payoff = payoff[x];
    double neig_payoff = payoff[neig];

    // Dynamically compute the noise parameter based on x's group size: K_actual = k_norm * G
    int G = local_group_size[x];
    double K_actual = k_norm * G;

    // Fermi function: prob = 1 / (1 + exp((x_payoff - neig_payoff) / K_actual))
    double prob = 1.0 / (1.0 + exp((x_payoff - neig_payoff) / K_actual));

    // With probability prob, adopt the neighbor's strategy
    if (randf() < prob) {
        strategy[x] = strategy[neig];
    }
}

// Execute one round of the game
void round_game() {
    // At the start of each round, reset all players' movement states
    for (int i = 0; i < N; i++) {
        has_moved[i] = false;
    }
    movable_moved_count = 0;

    // Store a list of players who need to move
    vector<int> players_to_move;

    // Traverse all occupied nodes to see who meets the condition to move
    for (int i : occupied_sites) {
        if (can_move(i)) {
            players_to_move.push_back(i);
        }
    }

    // Randomly shuffle the order of movement
    random_shuffle(players_to_move.begin(), players_to_move.end());

    // Move those players who meet the condition
    for (int i : players_to_move) {
        move_player(i);
    }

    // Calculate payoffs for all occupied nodes
    for (int i : occupied_sites) {
        // Only compute payoff for valid strategies (non-empty)
        if (strategy[i] == 0.0 || strategy[i] == 1.0) {
            payoff[i] = cal_payoff(i);
        }
    }

    // Strategy update phase: each player can learn from a neighbor
    vector<int> current_occupied = occupied_sites; // Copy the list of occupied nodes
    for (int i : current_occupied) {
        if (strategy[i] == 0.0 || strategy[i] == 1.0) {
            learn_strategy(i);
        }
    }
}

// Arrays for output data: store average cooperation rates and payoffs of different categories of players
double data_out[3];       // data_out[0]: cooperation rate of movable players, data_out[1]: non-movable, data_out[2]: overall
double data_payoff[3];    // data_payoff[0]: average payoff of movable players, data_payoff[1]: non-movable, data_payoff[2]: overall

// Compute the cooperation rate and payoff data for different categories of players in the current round
void cal_data() {
    int movable_cooperators = 0, movable_count = 0;
    int non_movable_cooperators = 0, non_movable_count = 0;
    int total_cooperators = 0, occupied_count = 0;

    double total_movable_payoff = 0.0;
    double total_non_movable_payoff = 0.0;
    double total_occupied_payoff = 0.0;

    // Traverse all occupied nodes, gathering info about movable and non-movable players
    for (int i : occupied_sites) {
        // Only consider valid strategies (non-empty nodes)
        if (strategy[i] == 1.0 || strategy[i] == 0.0) {
            if (is_movable[i]) {
                movable_count++;
                total_movable_payoff += payoff[i];
                if (strategy[i] == 1.0) {
                    movable_cooperators++;
                }
            } else {
                non_movable_count++;
                total_non_movable_payoff += payoff[i];
                if (strategy[i] == 1.0) {
                    non_movable_cooperators++;
                }
            }
            occupied_count++;
            total_occupied_payoff += payoff[i];

            if (strategy[i] == 1.0) {
                total_cooperators++;
            }
        }
    }

    // Compute the cooperation rate for each category
    data_out[0] = (movable_count > 0) ? ((double)movable_cooperators / movable_count) : 0.0;
    data_out[1] = (non_movable_count > 0) ? ((double)non_movable_cooperators / non_movable_count) : 0.0;
    data_out[2] = (occupied_count > 0) ? ((double)total_cooperators / occupied_count) : 0.0;

    // Compute the average payoff for each category
    data_payoff[0] = (movable_count > 0) ? (total_movable_payoff / movable_count) : 0.0;
    data_payoff[1] = (non_movable_count > 0) ? (total_non_movable_payoff / non_movable_count) : 0.0;
    data_payoff[2] = (occupied_count > 0) ? (total_occupied_payoff / occupied_count) : 0.0;
}

// Compute the standard deviation of the experimental results
void calculate_standard_deviation(const vector<double> &experiment_results, double mean, double &std_dev) {
    double variance = 0.0;
    // Compute the squared difference from the mean for each result
    for (double result : experiment_results) {
        variance += (result - mean) * (result - mean);
    }
    // Compute the mean squared difference
    variance /= experiment_results.size();
    // Standard deviation is the square root of the variance
    std_dev = sqrt(variance);
}

int main(void) {
    int Round = 100000;     // Number of rounds in each experiment
    int Experiments = 100; // Number of independent experiments

    sgenrand(time(0));   // Initialize the RNG seed with the current time

    printf("*****start*****\n");

    // Open a CSV file to save experimental results
    FILE *Fc = fopen("Migration+Interactive identity.csv", "w");
    if (Fc == NULL) {
        perror("Failed to open file");
        return EXIT_FAILURE;
    }

    // Write the header (title row) to the CSV file
    fprintf(Fc, "rho,r_norm,Final_Fc_m,Final_Fc_nm,Final_Fc,Std_Fc_m,Std_Fc_nm,Std_Fc,"
                "Final_Payoff_m,Final_Payoff_nm,Final_Payoff,Std_Payoff_m,Std_Payoff_nm,Std_Payoff\n");

    // Outer loop: iterate over different player densities rho, from 0.00 to 1.00 in steps of 0.01
    for (double rho = 0.10; rho <= 1.00; rho += 0.01) {
        // Inner loop: iterate over different normalized cooperation factors r_norm, from 0.00 to 1.50 in steps of 0.01
        for (r_norm = 0.00; r_norm <= 1.50; r_norm += 0.01) {
            double total_data[3] = { 0.0, 0.0, 0.0 };       // Accumulate cooperation rates for each category
            double total_payoff_data[3] = { 0.0, 0.0, 0.0 }; // Accumulate average payoff data for each category
            vector<double> exp_results[3];                  // Store cooperation rate results from each experiment
            vector<double> exp_payoff_results[3];           // Store payoff results from each experiment

            // Perform multiple independent experiments
            for (int exp = 0; exp < Experiments; exp++) {
                // Initialize the game based on the current density rho
                init_game(rho);

                // Accumulate the cooperation rates/payoffs for the last 2000 rounds
                double data_sum[3] = { 0.0, 0.0, 0.0 };
                double data_sum_payoff[3] = { 0.0, 0.0, 0.0 };

                // Run the simulation for Round rounds
                for (int i = 0; i < Round; i++) {
                    // Execute one round (movement, payoff calculation, strategy update)
                    round_game();
                    // Collect cooperation/payoff data
                    cal_data();

                    // Only gather data for the last 5000 rounds (for steady-state results)
                    if (i >= Round - 5000) {
                        for (int j = 0; j < 3; j++) {
                            data_sum[j] += data_out[j];
                            data_sum_payoff[j] += data_payoff[j];
                        }
                    }
                }

                // Compute average cooperation rate and payoff over the last 5000 rounds
                for (int j = 0; j < 3; j++) {
                    double avg_strategy = data_sum[j] / 5000.0;
                    total_data[j] += avg_strategy;
                    exp_results[j].push_back(avg_strategy);

                    double avg_payoff = data_sum_payoff[j] / 5000.0;
                    total_payoff_data[j] += avg_payoff;
                    exp_payoff_results[j].push_back(avg_payoff);
                }

                // Print progress and data of this experiment
                printf("rho = %.2f, r_norm = %.2f, Experiment = %d: Fc_m = %.4f, Fc_nm = %.4f, Fc = %.4f, "
                       "Payoff_m = %.4f, Payoff_nm = %.4f, Payoff = %.4f\n",
                       rho, r_norm, exp + 1,
                       data_sum[0] / 5000.0, data_sum[1] / 5000.0, data_sum[2] / 5000.0,
                       data_sum_payoff[0] / 5000.0, data_sum_payoff[1] / 5000.0, data_sum_payoff[2] / 5000.0);
            }

            // Calculate mean and standard deviation for each category
            double mean[3], std_dev[3];
            double mean_payoff[3], std_dev_payoff[3];
            for (int j = 0; j < 3; j++) {
                mean[j] = total_data[j] / Experiments;
                calculate_standard_deviation(exp_results[j], mean[j], std_dev[j]);

                mean_payoff[j] = total_payoff_data[j] / Experiments;
                calculate_standard_deviation(exp_payoff_results[j], mean_payoff[j], std_dev_payoff[j]);
            }

            // Write results for the current density/cooperation factor to the CSV file
            fprintf(Fc, "%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                        "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    rho, r_norm,
                    mean[0], mean[1], mean[2],
                    std_dev[0], std_dev[1], std_dev[2],
                    mean_payoff[0], mean_payoff[1], mean_payoff[2],
                    std_dev_payoff[0], std_dev_payoff[1], std_dev_payoff[2]);

            // Print the final statistical data for the current setting in the console
            printf("rho = %.2f, r_norm = %.2f: Final_Fc_m = %.4f, Final_Fc_nm = %.4f, Final_Fc = %.4f, "
                   "Final_Payoff_m = %.4f, Final_Payoff_nm = %.4f, Final_Payoff = %.4f\n",
                   rho, r_norm,
                   mean[0], mean[1], mean[2],
                   mean_payoff[0], mean_payoff[1], mean_payoff[2]);
        }
    }

    fclose(Fc); // Close the output file
    printf("*****done*****\n");
    return 0;
}
