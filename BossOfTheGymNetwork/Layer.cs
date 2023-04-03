using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    [Serializable]
    public class Layer
    {
        public int ID { get; set; }
        public List<Neuron> Neurons { get; set; }
        public int NeuronCount => Neurons?.Count ?? 0;
        public NeuronType NeuronType { get; }
        public Layer(List<Neuron> neurons,int id,NeuronType neuronType = NeuronType.Hidden)
        {
            Neurons = neurons;
            NeuronType = neuronType;
            ID = id;
        }
        public List<double> GetSignals()
        {
            var result = new List<double>();

            foreach(var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }
        public override string ToString() => NeuronType.ToString();
    }
}
