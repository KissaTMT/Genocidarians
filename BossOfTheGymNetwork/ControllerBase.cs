using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public abstract class  ControllerBase
    {
        private readonly IDataSaver manager = new SerializeDataSaver();

        protected void Save<T>(List<T> collection) where T : class => manager.Save(collection);
        protected List<T> Load<T>() where T : class => manager.Load<T>();
    }
}
