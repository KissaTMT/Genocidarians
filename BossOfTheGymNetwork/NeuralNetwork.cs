using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    [DataContract]
    public class NeuralNetwork : ControllerBase
    {
        public Topology Topology { get; }
        [DataMember] public List<Layer> Layers { get; set; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;

            Layers = new List<Layer>();
            CreateInputLayer();
            CreateHiddenLayers();
            CreateOututLayer();
        }

        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1) return Layers.Last().Neurons[0];
            else return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
        }

        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            //var signals = Normalization(inputs);
            var error = 0.0;

            for(var i = 0; i < epoch; i++)
            {
                for (var j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);
                    error += Backpropagation(output, input);
                }
            }
            Save(Layers);
            return error / epoch;
        }

        public static double[] GetRow(double [,] matrix, int row)
        {
            var column = matrix.GetLength(1);
            var array = new double[column];

            for(var i = 0; i < column; ++i)
            {
                array[i] = matrix[row, i];
            }

            return array;
        }

        private double [,] Normalization(double[,] inputs) 
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (var column = 0; column < inputs.GetLength(1); column++)
            {
                var sum = 0.0;
                var len = inputs.GetLength(0);
                for(var row = 1; row < len; row++)
                {
                    sum += inputs[row, column];
                }
                var average = sum / len;
                var error = 0.0;

                for(var row = 1; row < len; row++)
                {
                    error += Math.Pow(inputs[row, column] - average, 2);
                }
                var standartError = Math.Sqrt(error/len);
                for(var row = 1; row < len; row++)
                {
                    result[row,column] = (inputs[row, column] - average) / standartError;
                }
            }
            return result;
        }
        private double [,] Scalling(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for(var column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];

                for(var row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, column];

                    if (item < min) min = item;
                    if (item > max) max = item;
                }
                var delta = max - min;
                for (var row = 1; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - min) / delta;
                }
            }
            return result;
        }
        private double Backpropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;

            var difference = actual - expected;// error on output neuron

            foreach(var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }
            for(var i = Layers.Count - 2; i >= 0; i--)
            {
                var layer = Layers[i];
                var previousLayer = Layers[i + 1];

                for(var j = 0; j < layer.NeuronCount; j++)
                {
                    var neuron = layer.Neurons[j];
                    for(var k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];

                        var error = previousNeuron.Weights[j] * previousNeuron.Delta; // error on other neurons
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }
            return difference * difference;
        }
        private void FeedForwardAllLayersAfterInput()
        {
            for (var i = 1; i < Layers.Count; i++)
            {
                var previousLayerSignals = Layers[i - 1].GetSignals();
                var layer = Layers[i];

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (var i = 0; i < inputSignals.Length; i++)
            {
                var neuron = Layers[0].Neurons[i];
                neuron.FeedForward(new List<double>() { inputSignals[i] });
            }
        }

        private void CreateInputLayer()
        {
            var neurons = new List<Neuron>();
            for(var i = 0; i < Topology.InputCount; i++)
            {
                neurons.Add(new Neuron(1, NeuronType.Input));
            }
            Layers.Add(new Layer(neurons, -1, NeuronType.Input));
        }
        private void CreateOututLayer()
        {
            var layer = Load<Layer>();
            if (layer == null)
            {
                var neurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (var i = 0; i < Topology.OutputCount; i++)
                {
                    neurons.Add(new Neuron(lastLayer.NeuronCount, NeuronType.Output));
                }
                Layers.Add(new Layer(neurons, -1, NeuronType.Output));
            }
            else Layers.Add(layer.FirstOrDefault(i => i.NeuronType == NeuronType.Output));
        }
        private void CreateHiddenLayers()
        {
            for (var j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var layer = Load<Layer>();
                if (layer == null)
                {
                    var neurons = new List<Neuron>();
                    var lastLayer = Layers.Last();
                    for (var i = 0; i < Topology.HiddenLayers[j]; i++)
                    {
                        neurons.Add(new Neuron(lastLayer.NeuronCount));
                    }
                    Layers.Add(new Layer(neurons, j));
                }
                else Layers.Add(layer.FirstOrDefault(i => i.ID == j));
            }
        }
    }
}
