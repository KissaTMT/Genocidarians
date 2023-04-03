using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public enum NeuronType
    {
        Input,
        Hidden,
        Output
    }
    [Serializable]
    public class Neuron
    {
        public List<double> Weights { get; set; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output => _output;
        public double Delta => _delta;

        private double _output;
        private double _delta;
        public Neuron(int inputCount, NeuronType neuronType = NeuronType.Hidden)
        {
            NeuronType = neuronType;
            Weights = new List<double>();
            Inputs = new List<double>();
            InitializeWeightsRandomValues(inputCount);
        }
        public double FeedForward(List<double> inputs)
        {
            for(var i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for(var i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }
            if (NeuronType != NeuronType.Input) _output = Sigmoid(sum);
            else _output = sum;
            return _output;
        }
        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input) return;
            _delta = error * SigmoidDx(_output); //delta

            for(var i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i] - Inputs[i] * _delta * learningRate;
                Weights[i] = weight;
            }

        }
        public override string ToString() => _output.ToString();
        private void InitializeWeightsRandomValues(int inputCount)
        {
            var random = new Random();
            for (var i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input) Weights.Add(1);
                else Weights.Add(random.NextDouble());
                Inputs.Add(0);
            }
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Pow(Math.E, -x));
        private double SigmoidDx(double x) => Sigmoid(x) / (1 - Sigmoid(x));
    }
}
