import numpy as np
import matplotlib.pyplot as plt
import os

class DiscreteSignal:
    def __init__(self,INF):
        self.INF = INF
        self.values = np.zeros(2 * INF + 1)

    def set_value_at_time(self,time,value):
        self.values[self.INF + time] = value
    
    def shift_signal(self,shift):
        temp = DiscreteSignal(self.INF)
        if(shift > self.INF * 2 + 1 or shift < -(self.INF * 2 + 1)):
            return temp
        if(shift == 0):
            return self
        if(shift > 0):
            temp.values = np.concatenate((self.values,temp.values))
            temp.values = np.roll(temp.values,shift)
            temp.values = temp.values[:2 * self.INF + 1]
            return temp
        else:
            temp.values = np.concatenate((temp.values,self.values))
            temp.values = np.roll(temp.values,shift)
            temp.values = temp.values[2 * self.INF + 1:]
            return temp

    def add(self,other):
        result = DiscreteSignal(self.INF)
        result.values = self.values + other.values
        return result
    
    def multiply(self,other):
        result = DiscreteSignal(self.INF)
        result.values = self.values * other.values
        return result
    
    def multiply_const_factor(self,const):
        result = DiscreteSignal(self.INF)
        result.values = const * self.values
        return result
    
    def plot(self):
        plt.stem(np.arange(-self.INF,self.INF+1),self.values)
        plt.show()



class ContinuousSignal:
    def __init__(self,function):
        self.function = function
    
    def shift(self,shift):
        def shifted_function(t):
            return self.function(t-shift)
        signal = ContinuousSignal(shifted_function)
        return signal
    
    def add(self,other):
        if isinstance(other,ContinuousSignal):
            def added_function(t):
                return self.function(t) + other.function(t)
            signal = ContinuousSignal(added_function)
            return signal  
        else:
            print("The other object is not a ContinuousSignal object")
    
    def multiply(self,other):
        if isinstance(other,ContinuousSignal):
            def multiplied_function(t):
                return self.function(t) * other.function(t)
            signal = ContinuousSignal(multiplied_function)
            return signal
        else:
            print("The other object is not a ContinuousSignal object")
    
    def multiply_const_factor(self,const):
        def multiplied_function(t):
            return const * self.function(t)
        signal = ContinuousSignal(multiplied_function)
        return signal
    
    def plot(self):
        t = np.linspace(-10,10,10000)
        plt.plot(t,self.function(t))
        plt.show()


class LTI_Discrete:
    def __init__(self,impulse_response):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self,input_signal):
        impulses = []
        coefficients = []
        for index,value in enumerate(input_signal.values):
            if value != 0:
                impulse = DiscreteSignal(input_signal.INF)
                impulse.set_value_at_time(0,1)
                impulse.shift_signal(index - input_signal.INF)
                impulses.append(impulse)
                coefficients.append(value)
            else:
                impulse = DiscreteSignal(input_signal.INF)
                impulses.append(impulse)
                coefficients.append(0)
        return impulses,coefficients

    def output(self,input_signal):
        impulses,coefficients = self.linear_combination_of_impulses(input_signal)
        impulse_responses = []
        output_signal = DiscreteSignal(input_signal.INF)
        for i in range(-input_signal.INF,input_signal.INF+1):
            impulse_response = self.impulse_response.shift_signal(i)
            impulse_responses.append(impulse_response)
            output_signal = output_signal.add(impulse_response.multiply_const_factor(input_signal.values[i+input_signal.INF]))
        return output_signal,impulse_responses,coefficients

class LTI_Continuous:
    def __init__(self,impulse_response):
        self.impulse_response = impulse_response
    
    def linear_combination_of_impulses(self,input_signal,delta):
        impulses = []
        t_array = np.arange(-3,3,delta)
        coefficients = input_signal.function(t_array)
        for t in t_array:
            impulse = ContinuousSignal(lambda t: 1/delta * (t >=0) * (t < delta))
            impulse = impulse.shift(t)
            impulses.append(impulse)
        
        return impulses,coefficients

    def output_approx(self,input_signal,delta):
        impulses,coefficients = self.linear_combination_of_impulses(input_signal,delta)
        impulse_responses = []
        output_signal = ContinuousSignal(lambda t: 0)
        for i in np.arange(-3,3+1,delta):
            impulse_response = self.impulse_response.shift(i)
            impulse_responses.append(impulse_response)
            output_signal = output_signal.add(impulse_response.multiply_const_factor(input_signal.function(i) * delta))
        return output_signal,impulse_responses,coefficients


def discrete_main():
    INF = 5
    input_signal = DiscreteSignal(INF)
    input_signal.set_value_at_time(0,0.5)
    input_signal.set_value_at_time(1,2)

    impulse_response = DiscreteSignal(INF)
    impulse_response.set_value_at_time(0,1)
    impulse_response.set_value_at_time(1,1)
    impulse_response.set_value_at_time(2,1)

    lti_discrete = LTI_Discrete(impulse_response)
    impulses,coefficients = lti_discrete.linear_combination_of_impulses(input_signal)

    folder_path = "./discrete_plots"
    os.makedirs(folder_path, exist_ok=True) 

    file_path = os.path.join(folder_path, "impulses.png")

    plt.stem(np.arange(-INF,INF+1),input_signal.values)
    plt.title("x[n]")
    plt.xlabel("n(Time index)")
    plt.ylabel("x[n]")
    plt.ylim(-1,4)
    plt.savefig("./discrete_plots/input_signal.png")
    plt.close()

    plt.stem(np.arange(-INF,INF+1),impulse_response.values)
    plt.title("h[n]")
    plt.xlabel("n(Time index)")
    plt.ylabel("h[n]")
    plt.ylim(-1,4)
    plt.savefig("./discrete_plots/impulse_response.png")
    plt.close()

    fig, axs = plt.subplots(4, 3, figsize=(15, 15)) 
    fig.suptitle("Impulse multiplied by coefficients", fontsize=16)
    fig.text(0.5, 0.01, "Figure 2: Returned impulses multiplied by respective coefficients.", ha='center', fontsize=12)
    for i in range(4):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(impulses):  
                axs[i, j].stem(np.arange(-INF, INF + 1), impulses[idx].values * coefficients[idx])
                axs[i, j].set_ylim(-1, 5) 
                axs[i,j].set_xlabel("n(Time index)")
                axs[i,j].set_ylabel("x[n]")

    impulse_sum = DiscreteSignal(INF)
    for impulse,coefficient in zip(impulses,coefficients):
        impulse_sum = impulse_sum.add(impulse.multiply_const_factor(coefficient))
    axs[3, 2].stem(np.arange(-INF, INF + 1), impulse_sum.values)
    axs[3, 2].set_ylim(-1, 5)
    axs[3,2].set_xlabel("n(Time index)")
    axs[3,2].set_ylabel("x[n]")
    axs[3,2].set_title("Sum")
    plt.tight_layout() 
    plt.savefig(file_path)
    plt.close()

    output_signal,impulse_responses,coefficients = lti_discrete.output(input_signal)

    fig, axs = plt.subplots(4, 3, figsize=(15, 15)) 
    fig.suptitle("Response of input signal", fontsize=16)
    for i in range(4):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(impulse_responses):  
                axs[i, j].stem(np.arange(-INF, INF + 1), impulse_responses[idx].values * coefficients[idx])
                axs[i, j].set_ylim(-1, 5) 
                axs[i,j].set_xlabel("n(Time index)")
                axs[i,j].set_ylabel("x[n]")

    axs[3, 2].stem(np.arange(-INF, INF + 1), output_signal.values)
    axs[3, 2].set_ylim(-1, 5)
    axs[3,2].set_xlabel("n(Time index)")
    axs[3,2].set_ylabel("x[n]")
    plt.tight_layout() 
    plt.savefig("./discrete_plots/output_signal.png")
    plt.close()

def input_function(t):
    return np.exp(-t) * (t >= 0)

def impulse_response_function(t):
    return  1* (t>=0)

def output_function(t):
    return (t >= 0) * (1 - np.exp(-t))

def continuous_main():
    folder_path = "./continuous_plots"
    os.makedirs(folder_path, exist_ok=True) 
    INF = 3
    input_signal = ContinuousSignal(input_function)

    t = np.linspace(-3,3,1000)
    plt.plot(t,input_signal.function(t))
    plt.title("x(t),INF = 3")
    plt.xlabel("t(Time)")
    plt.ylabel("x(t)")
    plt.ylim(0,1.1)
    plt.savefig("./continuous_plots/input_signal.png")
    plt.close()

    impulse_response = ContinuousSignal(impulse_response_function)

    lti_continuous = LTI_Continuous(impulse_response)
    delta = 0.5
    impulses, coefficients = lti_continuous.linear_combination_of_impulses(input_signal,delta)


    file_path = os.path.join(folder_path, "impulses.png")

    fig, axs = plt.subplots(5, 3, figsize=(15, 15)) 
    fig.suptitle("Impulse multiplied by coefficients", fontsize=16)
    for i in range(4):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(impulses):  
                t = np.linspace(-3,3,1000)
                axs[i,j].plot(t,impulses[idx].function(t) * coefficients[idx] * delta)
                axs[i,j].set_ylim(-0.1, 1.1) 
                axs[i,j].set_xlabel("t(Time)")
                axs[i,j].set_ylabel("x(t)")

    impulse_sum = ContinuousSignal(lambda t: 0)
    for impulse,coefficient in zip(impulses,coefficients):
        impulse_sum = impulse_sum.add(impulse.multiply_const_factor(coefficient * delta))
    t = np.linspace(-3,3,1000)
    axs[4,0].plot(t,impulse_sum.function(t))
    axs[4,0].set_ylim(-0.1, 1.1)
    axs[4,0].set_xlabel("t(Time)")
    axs[4,0].set_ylabel("x(t)")
    axs[4,0].set_title("Sum")
    plt.savefig(file_path)
    plt.close()

    fig,axs = plt.subplots(2,2,figsize=(15,15))

    axs[0,0].plot(t,input_signal.function(t),label="x(t)",color="orange")
    axs[0,0].plot(t,impulse_sum.function(t),label="Reconstructed",color="blue")
    axs[0,0].set_title("delta = " + str(delta))
    axs[0,0].legend()
    axs[0,0].set_xlabel("t(Time)")
    axs[0,0].set_ylabel("x(t)")

    delta = 0.1

    impulses, coefficients = lti_continuous.linear_combination_of_impulses(input_signal,delta)
    impulse_sum = ContinuousSignal(lambda t: 0)
    for impulse,coefficient in zip(impulses,coefficients):
        impulse_sum = impulse_sum.add(impulse.multiply_const_factor(coefficient * delta))
    
    axs[0,1].plot(t,input_signal.function(t),label="x(t)",color="orange")
    axs[0,1].plot(t,impulse_sum.function(t),label="Reconstructed",color="blue")
    axs[0,1].set_title("delta = " + str(delta))
    axs[0,1].legend()
    axs[0,1].set_xlabel("t(Time)")
    axs[0,1].set_ylabel("x(t)")

    delta = 0.05
    impulses, coefficients = lti_continuous.linear_combination_of_impulses(input_signal,delta)
    impulse_sum = ContinuousSignal(lambda t: 0)
    for impulse,coefficient in zip(impulses,coefficients):
        impulse_sum = impulse_sum.add(impulse.multiply_const_factor(coefficient * delta))
    
    axs[1,0].plot(t,input_signal.function(t),label="x(t)",color="orange")
    axs[1,0].plot(t,impulse_sum.function(t),label="Reconstructed",color="blue")
    axs[1,0].set_title("delta = " + str(delta))
    axs[1,0].legend()
    axs[1,0].set_xlabel("t(Time)")
    axs[1,0].set_ylabel("x(t)")

    delta = 0.01

    impulses, coefficients = lti_continuous.linear_combination_of_impulses(input_signal,delta)
    impulse_sum = ContinuousSignal(lambda t: 0)
    for impulse,coefficient in zip(impulses,coefficients):
        impulse_sum = impulse_sum.add(impulse.multiply_const_factor(coefficient * delta))
    
    axs[1,1].plot(t,input_signal.function(t),label="x(t)",color="orange")
    axs[1,1].plot(t,impulse_sum.function(t),label="Reconstructed",color="blue")
    axs[1,1].set_title("delta = " + str(delta))
    axs[1,1].legend()
    axs[1,1].set_xlabel("t(Time)")
    axs[1,1].set_ylabel("x(t)")

    plt.savefig("./continuous_plots/reconstructed_signals.png")
    plt.close()

    delta = 0.5

    output_signal,impulse_responses,coefficients = lti_continuous.output_approx(input_signal,delta)

    fig, axs = plt.subplots(5, 3, figsize=(15, 15))
    fig.suptitle("Response of input signal", fontsize=16)
    for i in range(4):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(impulse_responses):  
                t = np.linspace(-3,3,1000)
                axs[i,j].plot(t,impulse_responses[idx].function(t) * coefficients[idx])
                axs[i,j].set_ylim(-0.1, 1.1) 
                axs[i,j].set_xlabel("t(Time)")
                axs[i,j].set_ylabel("x(t)")
    
    t = np.linspace(-3,3,1000)
    axs[4,0].plot(t,output_signal.function(t))
    axs[4,0].set_ylim(-0.1, 1.1)
    axs[4,0].set_xlabel("t(Time)")
    axs[4,0].set_ylabel("x(t)")
    axs[4,0].set_title("Sum")
    plt.savefig("./continuous_plots/output_signal_with_responses.png")
    plt.close()

    output_signal_exact = ContinuousSignal(output_function)
    
    t = np.linspace(-3,3,1000)

    fig,axs = plt.subplots(2,2,figsize=(15,15))
    fig.suptitle("Approximate outputs as delta tends to 0", fontsize=16)
    axs[0,0].plot(t,output_signal_exact.function(t),label="y(t)",color="orange")
    axs[0,0].plot(t,output_signal.function(t),label="y_approx(t)",color="blue")
    axs[0,0].set_title("delta = " + str(delta))
    axs[0,0].legend()
    axs[0,0].set_xlabel("t(Time)")
    axs[0,0].set_ylabel("y(t)")

    delta = 0.1

    output_signal,impulse_responses,coefficients = lti_continuous.output_approx(input_signal,delta)

    axs[0,1].plot(t,output_signal_exact.function(t),label="y(t)",color="orange")
    axs[0,1].plot(t,output_signal.function(t),label="y_approx(t)",color="blue")
    axs[0,1].set_title("delta = " + str(delta))
    axs[0,1].legend()
    axs[0,1].set_xlabel("t(Time)")
    axs[0,1].set_ylabel("y(t)")
    
    delta = 0.05

    output_signal,impulse_responses,coefficients = lti_continuous.output_approx(input_signal,delta)

    axs[1,0].plot(t,output_signal_exact.function(t),label="y(t)",color="orange")
    axs[1,0].plot(t,output_signal.function(t),label="y_approx(t)",color="blue")
    axs[1,0].set_title("delta = " + str(delta))
    axs[1,0].legend()
    axs[1,0].set_xlabel("t(Time)")
    axs[1,0].set_ylabel("y(t)")

    delta = 0.01

    output_signal,impulse_responses,coefficients = lti_continuous.output_approx(input_signal,delta)
    
    axs[1,1].plot(t,output_signal_exact.function(t),label="y(t)",color="orange")
    axs[1,1].plot(t,output_signal.function(t),label="y_approx(t)",color="blue")
    axs[1,1].set_title("delta = " + str(delta))
    axs[1,1].legend()
    axs[1,1].set_xlabel("t(Time)")
    axs[1,1].set_ylabel("y(t)")

    plt.savefig("./continuous_plots/output_signals.png")
    plt.close()

def main():
    discrete_main()
    continuous_main()

main()