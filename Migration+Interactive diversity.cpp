#include <iostream>     // Include iostream for standard input/output
#include <fstream>      // Include fstream for file read/write
#include <cstdlib>      // Include cstdlib for functions like rand(), srand()
#include <cstdio>       // Include cstdio for printf, fopen, etc.
#include <cmath>        // Include cmath for exp(), sqrt(), etc.
#include <ctime>        // Include ctime to get system time (for initializing random seed)
#include <vector>       // Include vector for dynamic arrays
#include <algorithm>    // Include algorithm for random_shuffle(), etc.

using namespace std;    // Use the std namespace

//--------------------------- 1. Constants and Global Variables ---------------------------//
#define L 400                // The lattice size
#define N (L * L)            // Total number of nodes in the lattice
#define neig_num 4           // Number of neighbors (up, down, left, right)

// Global containers and arrays for storing node information
vector<int> occupied_sites;  // Stores the indices of occupied nodes
vector<int> empty_sites;     // Stores the indices of empty nodes
int neighbors[N][neig_num];  // Stores 4 neighbors' indices for each node
double strategy[N];          // Stores the strategy value for each node (0~1 for cooperation probability, -1 for empty)
bool is_movable[N];          // Marks whether the player at each node can move
bool has_moved[N];           // Marks whether the player has already moved in the current round
double payoff[N];            // Stores accumulated payoffs for each player

//--------------------------- 2. Normalization Parameters ---------------------------//
double normalized_r;         // Global “normalized multiplier factor”; assigned in the main loop
double normalized_kappa = 0.1;  // A fixed normalized noise parameter normalized_kappa is used during the learning phase.

// In order to use “local group size G” in the learning phase, we introduce a global array to store
// “the group size G of the public goods game centered on node x.”
int local_group_size[N];

// Other game parameters
double c = 1.0;              // Cooperation cost
double f = 3.0;              // Movement cost

//--------------------------- 3. Mersenne Twister Random Number Generator ---------------------------//

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

// Random number generator buffer and index
static unsigned long mt_buffer[NN]; // Stores generated random numbers
static int mti = NN + 1;            // RNG index; NN+1 indicates not initialized

// Initialize the RNG with a seed
void sgenrand(unsigned long seed) {
    mt_buffer[0] = seed & 0xffffffff;
    for (mti = 1; mti < NN; mti++) {
        // Linear congruential generator update
        mt_buffer[mti] = (69069 * mt_buffer[mti - 1]) & 0xffffffff;
    }
}

// Generate a random number using Mersenne Twister
unsigned long genrand() {
    unsigned long y;
    static unsigned long mag01[2] = { 0x0, MATRIX_A };

    // If index exceeds range, regenerate random number series
    if (mti >= NN) {
        int kk;
        // If never initialized, use default seed
        if (mti == NN + 1) {
            sgenrand(4357);
        }
        // Update the first (NN - MM) numbers
        for (kk = 0; kk < NN - MM; kk++) {
            y = (mt_buffer[kk] & UPPER_MASK) | (mt_buffer[kk + 1] & LOWER_MASK);
            mt_buffer[kk] = mt_buffer[kk + MM] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        // Update the remaining part
        for (; kk < NN - 1; kk++) {
            y = (mt_buffer[kk] & UPPER_MASK) | (mt_buffer[kk + 1] & LOWER_MASK);
            mt_buffer[kk] = mt_buffer[kk + (MM - NN)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        // The last one
        y = (mt_buffer[NN - 1] & UPPER_MASK) | (mt_buffer[0] & LOWER_MASK);
        mt_buffer[NN - 1] = mt_buffer[MM - 1] ^ (y >> 1) ^ mag01[y & 0x1];
        mti = 0;
    }

    y = mt_buffer[mti++];
    // Tempering transformations
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);
    return y;
}

// Generate a random double in [0, 1)
double randf() {
    return ((double)genrand() * 2.3283064370807974e-10);
}

// Generate a random integer in [0, LIM-1]
long randi(unsigned long LIM) {
    return ((unsigned long)genrand() % LIM);
}

//--------------------------- 4. Initialize Neighbors ---------------------------//

// Initialize each node's neighbors in a periodic lattice
void find_neig() {
    for (int i = 0; i < N; i++) {
        neighbors[i][0] = i - L; // Up
        neighbors[i][1] = i + L; // Down
        neighbors[i][2] = i - 1; // Left
        neighbors[i][3] = i + 1; // Right

        // Periodic boundary conditions: top row, bottom row, left boundary, right boundary
        if (i < L) {
            neighbors[i][0] = i + L * (L - 1);
        }
        if (i >= L * (L - 1)) {
            neighbors[i][1] = i - L * (L - 1);
        }
        if (i % L == 0) {
            neighbors[i][2] = i + L - 1;
        }
        if (i % L == L - 1) {
            neighbors[i][3] = i - L + 1;
        }
    }
}

//--------------------------- 5. Global Counters ---------------------------//
// Used to track the number of different categories of players

int total_movable = 0;       // Total number of movable players
int total_non_movable = 0;   // Total number of non-movable players
int total_occupied = 0;      // Total number of occupied nodes
int movable_moved_count = 0; // Number of movable players who actually moved in this round

//--------------------------- 6. Game Initialization ---------------------------//
// Initialize the game state based on player density rho

void init_game(double rho) {
    find_neig(); // First initialize neighbors

    // Clear lists of occupied and empty nodes
    occupied_sites.clear();
    empty_sites.clear();

    total_movable = 0;
    total_non_movable = 0;
    total_occupied = 0;

    // First pass: decide whether each node is occupied
    for (int i = 0; i < N; i++) {
        if (randf() < rho) {                 // Occupy the node with probability rho
            is_movable[i] = (randf() < 0.5); // 50% chance to be “movable”
            occupied_sites.push_back(i);
            if (is_movable[i]) {
                total_movable++;
            } else {
                total_non_movable++;
            }
            total_occupied++;
        } else {
            // Mark empty node
            is_movable[i] = false;
            strategy[i] = -1.0;
            empty_sites.push_back(i);
        }
    }

    // Second pass: set the initial strategy for occupied nodes
    // We first look at its neighbors, then set initial strategy = (#cooperative neighbors)/(#neighbors).
    for (int i : occupied_sites) {
        int occupied_neighbors = 0;
        int coop_count = 0;

        // Count how many neighbors of i are occupied, and assume “50% chance that neighbor is cooperative”
        for (int j = 0; j < neig_num; j++) {
            int neig = neighbors[i][j];
            if (strategy[neig] != -1.0) {
                occupied_neighbors++;
                if (randf() < 0.5) {
                    coop_count++;
                }
            }
        }
        // If there are occupied neighbors, initial strategy = coop_count / occupied_neighbors; otherwise 0.5
        if (occupied_neighbors > 0) {
            strategy[i] = ((double)coop_count) / occupied_neighbors;
        } else {
            strategy[i] = 0.5;
        }
    }
}

//--------------------------- 7. Player Movement Logic ---------------------------//

// Check if player x needs to move
bool can_move(int x) {
    // If player is not movable, return false
    if (!is_movable[x]) return false;

    int occupied_neighbors = 0;
    double avg_strategy = 0.0;

    // Count how many neighbors of x are occupied, sum their strategies
    for (int i = 0; i < neig_num; i++) {
        int neig = neighbors[x][i];
        if (strategy[neig] != -1.0) {
            occupied_neighbors++;
            avg_strategy += strategy[neig];
        }
    }

    // If there are occupied neighbors, calculate average strategy; otherwise keep 0.0
    if (occupied_neighbors > 0) {
        avg_strategy /= occupied_neighbors;
    }

    // If there are no occupied neighbors, or the average neighbor strategy < 0.5, we decide to move
    return (occupied_neighbors == 0 || avg_strategy < 0.5);
}

// Move player x to a randomly chosen empty node
void move_player(int x) {
    // Only move if there is at least one empty node
    if (!empty_sites.empty()) {
        // Pick a random empty node index
        int index = randi(empty_sites.size());
        int new_pos = empty_sites[index]; // Target position

        // Transfer the player's strategy and “movable” state to the new position
        strategy[new_pos] = strategy[x];
        is_movable[new_pos] = is_movable[x];

        // Mark the old position as empty
        strategy[x] = -1.0;
        is_movable[x] = false;

        // Update the empty node list: replace this empty slot with the old position
        empty_sites[index] = x;

        // Update the occupied node list
        auto it = find(occupied_sites.begin(), occupied_sites.end(), x);
        if (it != occupied_sites.end()) {
            *it = new_pos;
        }

        // Mark that the player at new_pos has moved in this round
        has_moved[new_pos] = true;
        // Record +1 for the actual number of movable players who moved in this round
        movable_moved_count++;
    }
}

//--------------------------- 8. Payoff Calculation ---------------------------//
// Here strategy[x] in [0,1] represents the probability of cooperation;
// we use random coin flips to decide whether someone cooperates in the game.

double cal_payoff(int x) {
    double total_pay = 0.0;

    // --------------- (1) Public goods game centered on player x ---------------
    vector<int> group_members;
    group_members.push_back(x);

    // Collect its occupied neighbors
    for (int i = 0; i < neig_num; i++) {
        int neig = neighbors[x][i];
        if (strategy[neig] != -1.0) {
            group_members.push_back(neig);
        }
    }

    // If group size > 1, play the public goods game
    if (group_members.size() > 1) {
        int cooperators = 0;
        // Randomly decide if x cooperates
        bool x_cooperates = (randf() < strategy[x]);
        if (x_cooperates) cooperators++;

        // Other members
        for (int player : group_members) {
            if (player == x) continue;
            if (randf() < strategy[player]) {
                cooperators++;
            }
        }

        // Current group size G
        int G = (int)group_members.size();
        // Save the “group size centered on x” for noise usage in strategy updating
        local_group_size[x] = G;

        // Actual multiplier: actual_r = normalized_r * G
        double actual_r = normalized_r * G;
        // Total payoff = actual_r * cooperators
        double total_benefit = actual_r * cooperators;
        // x’s share = (total payoff / G) - c if x cooperated
        double payoff_x = (total_benefit / G);
        if (x_cooperates) {
            payoff_x -= c;
        }
        total_pay += payoff_x;
    }
    else {
        // If group size (x + neighbors) <= 1, no public goods payoff
        // Also store a minimal value to avoid problems in the learning phase
        local_group_size[x] = 1;
    }

    // --------------- (2) Public goods games centered on x's neighbors ---------------
    // x also participates in the neighbors' games, gaining or losing accordingly
    for (int i = 0; i < neig_num; i++) {
        int focal = neighbors[x][i];
        if (strategy[focal] == -1.0) continue; // If focal is empty, skip

        vector<int> group_focal;
        group_focal.push_back(focal);
        // Collect focal’s occupied neighbors
        for (int j = 0; j < neig_num; j++) {
            int neigh2 = neighbors[focal][j];
            if (strategy[neigh2] != -1.0) {
                group_focal.push_back(neigh2);
            }
        }

        if (group_focal.size() > 1) {
            int cooperators_focal = 0;
            // Decide if x cooperates in focal’s game
            bool x_cooperates_in_focal = (randf() < strategy[x]);
            if (x_cooperates_in_focal) {
                cooperators_focal++;
            }
            // Other members
            for (int player : group_focal) {
                if (player == x) continue;
                if (randf() < strategy[player]) {
                    cooperators_focal++;
                }
            }

            int G_focal = (int)group_focal.size();
            double actual_r_focal = normalized_r * G_focal;
            double total_benefit_focal = actual_r_focal * cooperators_focal;
            double payoff_x_focal = total_benefit_focal / G_focal;
            if (x_cooperates_in_focal) {
                payoff_x_focal -= c;
            }
            total_pay += payoff_x_focal;
        }
    }

    // If player x moved this round, subtract the movement cost f
    if (has_moved[x]) {
        total_pay -= f;
    }

    return total_pay;
}

//--------------------------- 9. Strategy Update ---------------------------//
// Learn strategy from a random neighbor via the Fermi function. Noise is scaled by local group size.

void learn_strategy(int x) {
    // Collect occupied neighbors
    vector<int> occupied_neighbors;
    for (int i = 0; i < neig_num; i++) {
        int neig = neighbors[x][i];
        if (strategy[neig] != -1.0) {
            occupied_neighbors.push_back(neig);
        }
    }
    // If x has no occupied neighbors, cannot learn
    if (occupied_neighbors.empty()) return;

    // Pick one neighbor at random
    int neig = occupied_neighbors[randi(occupied_neighbors.size())];

    double x_pay = payoff[x];
    double neig_pay = payoff[neig];

    // Use the public goods group size local_group_size[x]
    int G = local_group_size[x];
    // Actual noise = normalized_kappa * G
    double actual_kappa = normalized_kappa * G;

    // Fermi function: prob = 1 / (1 + exp((x_pay - neig_pay)/actual_kappa))
    double prob = 1.0 / (1.0 + exp((x_pay - neig_pay) / actual_kappa));

    // If random number < prob, x adopts the neighbor’s strategy
    if (randf() < prob) {
        strategy[x] = strategy[neig];
    }
}

//--------------------------- 10. Run One Round of the Game ---------------------------//
// Includes: player movement -> payoff calculation -> strategy update

void round_game() {
    // Reset movement state
    for (int i = 0; i < N; i++) {
        has_moved[i] = false;
    }
    movable_moved_count = 0;

    // Collect players who need to move
    vector<int> players_to_move;
    for (int i : occupied_sites) {
        if (can_move(i)) {
            players_to_move.push_back(i);
        }
    }
    // Shuffle the order of movement
    random_shuffle(players_to_move.begin(), players_to_move.end());

    // Perform movements
    for (int i : players_to_move) {
        move_player(i);
    }

    // Calculate payoffs for all occupied nodes
    for (int i : occupied_sites) {
        // If strategy is in [0,1], we have a valid player
        if (strategy[i] >= 0.0 && strategy[i] <= 1.0) {
            payoff[i] = cal_payoff(i);
        }
    }

    // Strategy update: copy occupied list first to avoid changes while updating
    vector<int> current_occ = occupied_sites;
    for (int i : current_occ) {
        if (strategy[i] != -1.0) {
            learn_strategy(i);
        }
    }
}

//--------------------------- 11. Data Statistics ---------------------------//
// data_out stores cooperation rates, data_payoff stores payoffs

double data_out[3];
double data_payoff[3];

// Compute the average strategy and payoff for different categories of players
// (movable, non-movable, overall)
void cal_data() {
    double total_movable_str = 0.0;
    double total_nonmov_str = 0.0;
    double total_all_str = 0.0;

    double total_movable_pay = 0.0;
    double total_nonmov_pay = 0.0;
    double total_all_pay = 0.0;

    int movable_count = 0;
    int nonmov_count = 0;
    int occ_count = 0;

    // Iterate over all occupied nodes
    for (int i : occupied_sites) {
        // If strategy in [0,1], this node is occupied
        if (strategy[i] >= 0.0 && strategy[i] <= 1.0) {
            // Count total occupied
            occ_count++;
            total_all_str += strategy[i];
            total_all_pay += payoff[i];

            // Check if movable
            if (is_movable[i]) {
                movable_count++;
                total_movable_str += strategy[i];
                total_movable_pay += payoff[i];
            } else {
                nonmov_count++;
                total_nonmov_str += strategy[i];
                total_nonmov_pay += payoff[i];
            }
        }
    }

    // Compute cooperation rate (average strategy)
    data_out[0] = (movable_count > 0) ? (total_movable_str / movable_count) : 0.0;
    data_out[1] = (nonmov_count > 0) ? (total_nonmov_str / nonmov_count) : 0.0;
    data_out[2] = (occ_count > 0) ? (total_all_str / occ_count) : 0.0;

    // Compute average payoff
    data_payoff[0] = (movable_count > 0) ? (total_movable_pay / movable_count) : 0.0;
    data_payoff[1] = (nonmov_count > 0) ? (total_nonmov_pay / nonmov_count) : 0.0;
    data_payoff[2] = (occ_count > 0) ? (total_all_pay / occ_count) : 0.0;
}

//--------------------------- 12. Standard Deviation Calculation ---------------------------//
// Given multiple experimental results experiment_results and their mean, compute std_dev

void calculate_standard_deviation(const vector<double> &experiment_results, double mean, double &std_dev) {
    double variance = 0.0;
    for (double result : experiment_results) {
        double diff = (result - mean);
        variance += diff * diff;
    }
    variance /= experiment_results.size();
    std_dev = sqrt(variance);
}

//--------------------------- 13. Main Function ---------------------------//

int main(void) {
    int Round = 100000;      // Number of rounds in each experiment
    int Experiments = 100;    // Number of independent experiments

    // Initialize the random seed
    sgenrand(time(0));

    printf("***** Start Simulation *****\n");

    // Open a CSV file to record experimental results
    FILE *Fc = fopen("Migration+Interactive diversity.csv", "w");
    if (Fc == NULL) {
        perror("Failed to open file");
        return EXIT_FAILURE;
    }

    // Write the header to the CSV file
    fprintf(Fc,
            "rho,normalized_r,"
            "Final_Fc_m,Final_Fc_nm,Final_Fc,"
            "Std_Fc_m,Std_Fc_nm,Std_Fc,"
            "Final_Payoff_m,Final_Payoff_nm,Final_Payoff,"
            "Std_Payoff_m,Std_Payoff_nm,Std_Payoff\n"
    );

    // Outer loop: player density rho from 0.10 to 1.00 in steps of 0.01
    for (double rho = 0.10; rho <= 1.00; rho += 0.01) {
        // Inner loop: normalized_r from 0.00 to 1.50 in steps of 0.01 (as per your code)
        for (double norm_r = 0.00; norm_r <= 1.50; norm_r += 0.01) {

            // Assign the global normalized_r
            normalized_r = norm_r;

            // For accumulating multiple experiments' cooperation rates and payoffs
            double total_data[3] = { 0.0, 0.0, 0.0 };
            double total_payoff_data[3] = { 0.0, 0.0, 0.0 };

            // Store each experiment's results for standard deviation
            vector<double> exp_results[3];
            vector<double> exp_payoff_results[3];

            // Multiple independent experiments
            for (int exp = 0; exp < Experiments; exp++) {
                // Initialize the game
                init_game(rho);

                // For accumulating “last few rounds” data
                double data_sum[3] = { 0.0, 0.0, 0.0 };
                double data_sum_payoff[3] = { 0.0, 0.0, 0.0 };

                // Run Round rounds
                for (int i = 0; i < Round; i++) {
                    // One round: movement -> payoff calculation -> strategy update
                    round_game();
                    // Compute data
                    cal_data();

                    // Only record data for the last 5000 rounds
                    if (i >= Round - 5000) {
                        for (int j = 0; j < 3; j++) {
                            data_sum[j] += data_out[j];
                            data_sum_payoff[j] += data_payoff[j];
                        }
                    }
                }

                // Compute average cooperation rate and payoff for this experiment
                for (int j = 0; j < 3; j++) {
                    double avg_fc = data_sum[j] / 5000.0;
                    double avg_pay = data_sum_payoff[j] / 5000.0;

                    // Add to totals
                    total_data[j] += avg_fc;
                    total_payoff_data[j] += avg_pay;

                    // Push into vectors for standard deviation
                    exp_results[j].push_back(avg_fc);
                    exp_payoff_results[j].push_back(avg_pay);
                }

                // Print progress info for each experiment
                printf("rho = %.2f, normalized_r = %.2f, Experiment = %d: Fc_m = %.4f, Fc_nm = %.4f, Fc = %.4f, "
                       "Payoff_m = %.4f, Payoff_nm = %.4f, Payoff = %.4f\n",
                       rho, norm_r, exp + 1,
                       (data_sum[0]/5000.0), (data_sum[1]/5000.0), (data_sum[2]/5000.0),
                       (data_sum_payoff[0]/5000.0), (data_sum_payoff[1]/5000.0), (data_sum_payoff[2]/5000.0));
            }

            // Calculate average and standard deviation across multiple experiments
            double mean_fc[3], std_fc[3];
            double mean_pay[3], std_pay[3];

            for (int j = 0; j < 3; j++) {
                // Mean
                mean_fc[j] = total_data[j] / Experiments;
                mean_pay[j] = total_payoff_data[j] / Experiments;
                // Standard deviation
                calculate_standard_deviation(exp_results[j], mean_fc[j], std_fc[j]);
                calculate_standard_deviation(exp_payoff_results[j], mean_pay[j], std_pay[j]);
            }

            // Write results to CSV
            fprintf(Fc, "%.2f,%.2f,"
                        "%.4f,%.4f,%.4f,"
                        "%.4f,%.4f,%.4f,"
                        "%.4f,%.4f,%.4f,"
                        "%.4f,%.4f,%.4f\n",
                    rho, norm_r,
                    mean_fc[0], mean_fc[1], mean_fc[2],
                    std_fc[0], std_fc[1], std_fc[2],
                    mean_pay[0], mean_pay[1], mean_pay[2],
                    std_pay[0], std_pay[1], std_pay[2]
            );

            // Print summary in console
            printf("rho = %.2f, normalized_r = %.2f: Final_Fc_m = %.4f, Final_Fc_nm = %.4f, Final_Fc = %.4f, "
                   "Final_Payoff_m = %.4f, Final_Payoff_nm = %.4f, Final_Payoff = %.4f\n",
                   rho, norm_r,
                   mean_fc[0], mean_fc[1], mean_fc[2],
                   mean_pay[0], mean_pay[1], mean_pay[2]
            );
        }
    }

    // Close file
    fclose(Fc);
    printf("***** Simulation Complete *****\n");
    return 0;
}