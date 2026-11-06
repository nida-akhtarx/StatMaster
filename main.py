
import streamlit as st
import statistics
import math
import random
from collections import Counter
from abc import ABC, abstractmethod

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="StatMaster",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# OOP: DATASET CLASS
# ==========================================

class Dataset:

    def __init__(self, data):
        self.data = data
        self.__std_dev = (
            statistics.stdev(data) if len(data) > 1 else 0
        )

    # Encapsulation: private standard deviation
    @property
    def std_dev(self):
        return self.__std_dev

    def mean(self):
        return statistics.mean(self.data)

    def median(self):
        return statistics.median(self.data)

    def mode(self):
        return statistics.multimode(self.data)

    def minimum(self):
        return min(self.data)

    def maximum(self):
        return max(self.data)

    def frequency(self):
        return dict(sorted(Counter(self.data).items()))

    def outliers(self):
        sorted_data = sorted(self.data)
        if len(sorted_data) < 4:
            return []

        q1 = statistics.median(sorted_data[:len(sorted_data)//2])
        mid = len(sorted_data) // 2

        if len(sorted_data) % 2 == 0:
            upper = sorted_data[mid:]
        else:
            upper = sorted_data[mid+1:]

        q3 = statistics.median(upper)
        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        # If-else outlier detection
        result = []
        for number in self.data:
            if number < lower_limit or number > upper_limit:
                result.append(number)
            else:
                continue

        return result


# ==========================================
# ABSTRACTION AND INHERITANCE
# ==========================================

class Distribution(ABC):

    @abstractmethod
    def calculate_probability(self):
        pass


class NormalDistribution(Distribution):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def calculate_probability(self, x):
        if self.std <= 0:
            return None

        z = (x - self.mean) / self.std
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class BinomialDistribution(Distribution):

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def calculate_probability(self, x):
        if x < 0 or x > self.n:
            return 0

        return (
            math.comb(self.n, x)
            * self.p**x
            * (1-self.p)**(self.n-x)
        )


# ==========================================
# HYPOTHESIS TESTING ENGINE
# ==========================================

# Fixed critical Z-values stored in a tuple
CRITICAL_Z = (
    (0.10, 1.645),
    (0.05, 1.96),
    (0.01, 2.576)
)


def hypothesis_decision(test_statistic, critical_value, alpha):
    if abs(test_statistic) > critical_value:
        return "Reject"
    else:
        return "Fail to Reject"


def run_hypothesis_test(data, test_type, population_mean, alpha):

    n = len(data)
    sample_mean = statistics.mean(data)

    match test_type:

        case "Z-Test":
            population_std = st.session_state.get("population_std", 0)

            if population_std <= 0:
                return None, None, "Enter a valid population standard deviation."

            z = (
                (sample_mean - population_mean)
                / (population_std / math.sqrt(n))
            )

            critical = next(
                (value for level, value in CRITICAL_Z
                 if level == alpha), 1.96
            )

            return z, critical, hypothesis_decision(z, critical, alpha)

        case "T-Test":
            if n < 2:
                return None, None, "At least two values are required."

            std = statistics.stdev(data)

            if std == 0:
                return None, None, "Sample standard deviation is zero."

            t = (
                (sample_mean - population_mean)
                / (std / math.sqrt(n))
            )

            # Approximate critical t-value
            critical = 1.96 + (2.5 / max(n - 1, 1))

            return t, critical, hypothesis_decision(t, critical, alpha)

        case "Chi-Square":
            observed = list(Counter(data).values())
            total = sum(observed)
            k = len(observed)

            if k < 2:
                return None, None, "At least two categories are required."

            expected = total / k

            chi_square = 0
            for value in observed:
                chi_square += ((value - expected) ** 2) / expected

            # Approximate critical value
            chi_critical_table = {
                1: (2.706, 3.841, 6.635),
                2: (4.605, 5.991, 9.210),
                3: (6.251, 7.815, 11.345),
                4: (7.779, 9.488, 13.277),
                5: (9.236, 11.070, 15.086),
                6: (10.645, 12.592, 16.812),
                7: (12.017, 14.067, 18.475),
                8: (13.362, 15.507, 20.090),
                9: (14.684, 16.919, 21.666),
                10: (15.987, 18.307, 23.209)
            }

            df = k - 1
            alpha_index = {0.10: 0, 0.05: 1, 0.01: 2}[alpha]

            if df not in chi_critical_table:
                return None, None, "Chi-square table supports up to 10 degrees of freedom."

            critical = chi_critical_table[df][alpha_index]

            if chi_square > critical:
                decision = "Reject"
            else:
                decision = "Fail to Reject"

            return chi_square, critical, decision

        case _:
            return None, None, "Invalid test selected."


# ==========================================
# RANDOM SAMPLING SIMULATION
# ==========================================

def simulate_sampling(trials):
    count = 0
    total = 0

    # While loop for random sampling trials
    while count < trials:
        number = random.randint(1, 6)
        if number == 6:
            total += 1
        count += 1

    return total / trials


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("📊 StatMaster")
st.sidebar.caption("Interactive Data & Probability Suite")

page = st.sidebar.selectbox(
    "Navigate",
    [
        "Descriptive Statistics",
        "Probability Distributions",
        "Hypothesis Testing"
    ]
)

# Match case navigation
match page:

    case "Descriptive Statistics":
        st.title("📊 Descriptive Statistics")
        st.write("Analyze your numerical dataset.")

        raw_data = st.text_input(
            "Enter numbers separated by commas",
            placeholder="10, 20, 30, 40, 50"
        )

        if raw_data:
            try:
                numbers = [
                    float(x.strip())
                    for x in raw_data.split(",")
                    if x.strip()
                ]

                if not numbers:
                    st.warning("Please enter some numbers.")

                elif len(numbers) < 1:
                    st.warning("Dataset is empty.")

                else:
                    dataset = Dataset(numbers)

                    st.subheader("Statistical Results")

                    c1, c2, c3 = st.columns(3)

                    c1.metric("Mean", f"{dataset.mean():.2f}")
                    c2.metric("Median", f"{dataset.median():.2f}")
                    c3.metric("Mode", str(dataset.mode()))

                    c4, c5, c6 = st.columns(3)

                    c4.metric("Minimum", dataset.minimum())
                    c5.metric("Maximum", dataset.maximum())
                    c6.metric("Standard Deviation", f"{dataset.std_dev:.2f}")

                    st.divider()

                    st.subheader("Frequency Distribution")

                    frequency = dataset.frequency()
                    st.bar_chart(frequency)

                    st.subheader("Outlier Detection")

                    outliers = dataset.outliers()

                    if outliers:
                        st.warning(f"Outliers detected: {outliers}")
                    else:
                        st.success("No outliers detected.")

                    st.subheader("Frequency Table")

                    st.dataframe(
                        {
                            "Number": list(frequency.keys()),
                            "Frequency": list(frequency.values())
                        },
                        use_container_width=True
                    )

            except ValueError:
                st.error("Invalid input! Enter numbers separated by commas.")

    # ======================================
    # PROBABILITY DISTRIBUTIONS
    # ======================================

    case "Probability Distributions":

        st.title("🎲 Probability Distributions")
        st.write("Explore Normal and Binomial distributions.")

        distribution_type = st.selectbox(
            "Select Distribution",
            ["Normal Distribution", "Binomial Distribution"]
        )

        if distribution_type == "Normal Distribution":

            st.subheader("Normal Distribution")

            mean = st.number_input("Enter Mean", value=50.0)
            std = st.number_input("Enter Standard Deviation", value=10.0)
            x = st.number_input("Enter X value", value=60.0)

            if st.button("Calculate Probability"):

                normal = NormalDistribution(mean, std)
                probability = normal.calculate_probability(x)

                if probability is None:
                    st.error("Standard deviation must be greater than zero.")
                else:
                    st.metric(
                        "Probability P(X ≤ x)",
                        f"{probability:.4f}"
                    )

                    st.write(
                        "This represents the cumulative probability "
                        "up to the selected X value."
                    )

        else:

            st.subheader("Binomial Distribution")

            n = st.number_input("Number of Trials (n)", 1, 1000, 10)
            p = st.number_input(
                "Probability of Success (p)",
                min_value=0.0,
                max_value=1.0,
                value=0.5
            )
            x = st.number_input(
                "Number of Successes (x)",
                min_value=0,
                max_value=1000,
                value=5
            )

            if st.button("Calculate Binomial Probability"):

                if x > n:
                    st.error("Successes cannot exceed total trials.")
                else:
                    binomial = BinomialDistribution(n, p)
                    result = binomial.calculate_probability(x)

                    st.metric("P(X = x)", f"{result:.6f}")

        st.divider()

        st.subheader("Random Sampling Simulation")

        trials = st.number_input(
            "Number of Trials",
            min_value=1,
            max_value=100000,
            value=1000
        )

        if st.button("Run Simulation"):
            result = simulate_sampling(int(trials))

            st.metric(
                "Experimental Probability of Rolling a 6",
                f"{result:.4f}"
            )

            st.write("Theoretical Probability: 0.1667")

    # ======================================
    # HYPOTHESIS TESTING
    # ======================================

    case "Hypothesis Testing":

        st.title("🧪 Hypothesis Testing")

        st.write(
            "Test a statistical hypothesis using Z-Test, "
            "T-Test, or Chi-Square."
        )

        test_type = st.selectbox(
            "Select Statistical Test",
            ["Z-Test", "T-Test", "Chi-Square"]
        )

        alpha = st.selectbox(
            "Significance Level (Alpha)",
            [0.10, 0.05, 0.01],
            index=1
        )

        st.info(f"Significance level: {alpha}")

        raw_data = st.text_input(
            "Enter Sample Data",
            placeholder="10, 12, 14, 16, 18"
        )

        population_mean = st.number_input(
            "Hypothesized Population Mean",
            value=15.0
        )

        if test_type == "Z-Test":
            population_std = st.number_input(
                "Population Standard Deviation",
                min_value=0.0,
                value=5.0
            )
            st.session_state["population_std"] = population_std

        if st.button("Perform Hypothesis Test"):

            try:
                data = [
                    float(x.strip())
                    for x in raw_data.split(",")
                    if x.strip()
                ]

                if len(data) < 2:
                    st.error("Enter at least two sample values.")

                else:
                    statistic_value, critical_value, decision = (
                        run_hypothesis_test(
                            data,
                            test_type,
                            population_mean,
                            alpha
                        )
                    )

                    if statistic_value is None:
                        st.error(decision)

                    else:
                        st.subheader("Test Results")

                        c1, c2 = st.columns(2)

                        c1.metric(
                            "Test Statistic",
                            f"{statistic_value:.4f}"
                        )

                        c2.metric(
                            "Critical Value",
                            f"{critical_value:.4f}"
                        )

                        st.subheader("Final Decision")

                        if decision == "Reject":
                            st.error(
                                "Reject the Null Hypothesis (H₀)"
                            )
                        else:
                            st.success(
                                "Fail to Reject the Null Hypothesis (H₀)"
                            )

                        st.write(
                            "Decision is based on comparing the "
                            "absolute test statistic with the "
                            "critical value."
                        )

            except ValueError:
                st.error("Please enter valid numerical data.")


# ==========================================
# FOOTER
# ==========================================

st.sidebar.divider()
st.sidebar.caption("StatMaster | Python Capstone Project")
st.sidebar.caption("Statistics & Probability Suite")
