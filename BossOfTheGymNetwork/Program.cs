using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
    public class Program
    {
        private static double[] kisa = { 1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,0,1 };
        //private static double[] pyshpysh = { 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0 };
        private static string[] qualityes =
        {
            "Авантюризм",
            "Авторитарность",
            "Отсут. Агрессивности",
            "Социальность",
            "Благородство",
            "Спокойствие",
            "Важность",
            "Весёлость",
            "Вежливость",
            "Воля",
            "Смелость",
            "Отзывчивость",
            "Искренность",
            "Геданизм",
            "Гениальность",
            "Интелект",
            "Гуманность",
            "Человечность",
            "Отсут. Двуличия",
            "Отсут. Дипрессии",
            "Дисциплинированность",
            "Доброта",
            "Отсут. Жестокости",
            "Заботливость",
            "Отсут. Капризности",
            "Отсут. Склонности к конфликтам",
            "Отсут. коварства",
            "Лидерство",
            "Честность",
            "Эмпатия",
            "Сочувствие",
            "Психологическое здоровье",
            "Идейность"
        };
        private static void Main(string[] args)
        {
            Console.WriteLine("Введите 'w' для работы с нейронкой");
            Console.WriteLine("Введите 'l' для обучения нейронки");

            var inputsList = new List<double[]>();
            var names = new List<string>();
            var outputs = new List<double>();


            var key = Console.ReadLine();

            var secondKey = string.Empty;
            if (key == "l")
            {
                var tuple = GetInputsToLearn();
                inputsList = tuple.inputs;
                outputs = tuple.outputs;
            }
            else if (key == "w")
            {
                Console.WriteLine("Введите 'cp' для копирования готовой(ых) последовательности(ей)");
                Console.WriteLine("Введите 'i' для ввода психологического образа");
                Console.WriteLine("Введите 'l' для загрузки данных из файла");
                secondKey = Console.ReadLine();
                if (secondKey == "cp") inputsList = GetInputsToWork();
                else if (secondKey == "i") inputsList.Add(GetInputToWork());
                else if (secondKey == "l")
                {
                    var tuple = LoadInputs();
                    names = tuple.names;
                    inputsList = tuple.inputs;
                }
                else return;
            }
            else return;

            var inputs = new double[inputsList.Count, inputsList[0].Length];

            for (var i = 0; i < inputs.GetLength(0); i++)
            {
                for (var j = 0; j < inputs.GetLength(1); j++)
                {
                    inputs[i, j] = inputsList[i][j];
                }
            }

            var len = inputs.GetLength(1);
            var topology = new Topology(len, 1, 0.005, len / 2 + 1);
            var neuralNetwork = new NeuralNetwork(topology);
            if (key == "l")
            {
                neuralNetwork.Learn(outputs.ToArray(), inputs, 1000000);
                for (var i = 0; i < outputs.Count; i++)
                {
                    Console.WriteLine("{0,-2} | {1} | {2,-10}", i + 2, outputs[i], string.Format("{0:P10}", neuralNetwork.FeedForward(NeuralNetwork.GetRow(inputs, i)).Output));
                }
            }
            if (key == "w")
            {
                if (secondKey == "l")
                {
                    var length = inputs.GetLength(0);
                    for (var i = 0; i < length; i++)
                    {
                        var someone = NeuralNetwork.GetRow(inputs, i);
                        var tuple = ComplexSequenceCompare(someone);
                        Console.WriteLine("{0,12} | {1,15} | {2,-2}/{3,-2} | {4,-2} | {5,8}", names[i], string.Format("{0:P10}", neuralNetwork.FeedForward(someone).Output), tuple.counterOfOne, someone.Length, tuple.counterPairsOfOne, string.Format("{0:P2}", tuple.counterProbability));
                    }
                    Console.WriteLine();
                    for(var i=0;i< length; i++)
                    {
                        var someone = NeuralNetwork.GetRow(inputs, i);
                        var tuple = ComplexSequenceCompare(someone);
                        Console.WriteLine("{0,12} | {1,15} | {2,-2}/{3,-2} | {4,-2} | {5,8}", names[i], string.Format("{0:P10}", neuralNetwork.FeedForward(someone).Output), tuple.counterOfOne, someone.Length, tuple.counterPairsOfOne, string.Format("{0:P2}", tuple.counterProbability));
                        for (var j = 0; j < qualityes.Length; j++)
                        {
                            Console.WriteLine("{0,30} | {1,1}", qualityes[j], someone[j]);
                        }
                        Console.WriteLine();
                    }
                }
                else
                {
                    for (var i = 0; i < inputs.GetLength(0); i++)
                    {
                        var someone = NeuralNetwork.GetRow(inputs, i);
                        var tuple = ComplexSequenceCompare(someone);
                        Console.WriteLine("{0,-3} | {1,15} | {2,-2}/{3,-2} | {4,-2} | {5,8}", i + 102, string.Format("{0:P10}", neuralNetwork.FeedForward(someone).Output), tuple.counterOfOne, someone.Length, tuple.counterPairsOfOne, string.Format("{0:P2}", tuple.counterProbability));
                    }
                }
            }
            Console.ReadLine();
        }
        private static (List<string> names, List<double[]> inputs) LoadInputs()
        {
            var inputs = new List<double[]>();
            var names = new List<string>();
            using(var sr = new StreamReader("protagonistsNames.txt"))
            {
                while (!sr.EndOfStream)
                {
                    names.Add(sr.ReadLine());
                }
            }
            using(var sr = new StreamReader("protagonistsData.txt"))
            {
                while (!sr.EndOfStream)
                {
                    var str = sr.ReadLine();
                    var len = str.Length;
                    var signals = new double[len];
                    for(var i = 0; i < len; i++)
                    {
                        signals[i] = double.Parse(str[i].ToString());
                    }
                    inputs.Add(signals);
                }
            }
            return (names, inputs);
        }
        private static (List<double[]> inputs, List<double> outputs) GetInputsToLearn()
        {
            var inputs = new List<double[]>();
            var outputs = new List<double>();
            while (true)
            {
                var str = Console.ReadLine();

                if (str == "e") break;

                outputs.Add(double.Parse(str[0].ToString()));


                var signals = new double[str.Length - 1];

                for (var i = 1; i < str.Length; i++)
                {
                    signals[i - 1] = double.Parse(str[i].ToString());
                }
                inputs.Add(signals);
            }
            return (inputs, outputs);
        }
        private static double[] GetInputToWork()
        {
            var len = qualityes.Length;
            var array = new double[len];

            Console.WriteLine("Введите имя: ");
            var name = Console.ReadLine();
            Console.WriteLine("Введите 0 или 1:");
            for (var i = 0; i < len; i++)
            {
                Console.WriteLine($"{qualityes[i]}:");
                while (true)
                {
                    var str = Console.ReadLine();
                    if (double.TryParse(str, out double value) && (value == 0 || value == 1))
                    {
                        array[i] = value;
                        break;
                    }
                    else Console.WriteLine("Неверный формат данных, попробуйте ещё раз");
                }
            }
            for (var i = 0; i < len; i++)
            {
                Console.Write(array[i]);
            }


            using (var sw = new StreamWriter("protagonistsNames.txt", true))
            {
                sw.WriteLine(name);
            }
            using (var sw = new StreamWriter("protagonistsData.txt", true))
            {
                for(var i = 0; i < len; i++)
                {
                    sw.Write(array[i]);
                }
                sw.WriteLine();
            }

            Console.WriteLine();
            return array;
        }
        private static List<double[]> GetInputsToWork()
        {
            var inputs = new List<double[]>();
            while (true)
            {
                var str = Console.ReadLine();

                if (str == "e") break;

                var signals = new double[str.Length];

                for (var i = 0; i < signals.Length; i++)
                {
                    signals[i] = double.Parse(str[i].ToString());
                }
                inputs.Add(signals);
            }
            Console.WriteLine("Сохранить в файл? y/n");
            var key = Console.ReadLine();
            if (key == "y")
            {
                Console.WriteLine("Введиье имя: ");
                var name = Console.ReadLine();
                using (var sw = new StreamWriter("protagonistsNames.txt", true))
                {
                    sw.WriteLine(name);
                }
                using (var sw = new StreamWriter("protagonistsData.txt", true))
                {
                    for (var i = 0; i < inputs[0].Length; i++)
                    {
                        sw.Write(inputs[0][i]);
                    }
                    sw.WriteLine();
                }
            }
            return inputs;
        }
        private static (int counterOfOne, int counterPairsOfOne, double counterProbability) ComplexSequenceCompare(double[] someone)
        {
            var counterProbability = 0.0;
            var counterOfOne = 0;
            var counterPairsOfOne = 0;

            for (var i = 0; i < someone.Length; i++)
            {
                if (kisa[i] == someone[i])
                {
                    counterProbability++;
                    if (someone[i] == 1) counterPairsOfOne++;
                }
                if (someone[i] == 1) counterOfOne++;
            }
            return (counterOfOne, counterPairsOfOne, counterProbability / someone.Length);
        }
    }
}