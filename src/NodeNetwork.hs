{-#LANGUAGE BangPatterns #-}
module NodeNetwork (NodeNetwork, netGen, evalNodeNetwork, cost, trainNetwork, keepEvalNodeNetwork) where

import Data.Foldable
import Control.Applicative
import Data.List
import System.Random
import Debug.Trace
import Data.Matrix

--the weights inside a node
type Weights a = [a]
--the inputs that go into a node
type Inputs a = [a]
--the outputs of a _layer_ of nodes
type Outputs a = [a]

defaultBias :: (Floating a, Show a) => a
defaultBias = (0)

--the node or neuron
--data Node a = Node {weights :: (Weights a), bias :: a} deriving (Show)
type Node a = Weights a

--a layer is a group of nodes
--All nodes in a layer must have the same number of inputs
--The number of nodes in a layer determines the number of outputs
type NodeLayer a = [Node a]

--a network is a group of layers
--The leftmost layer is the first
--The outputs of one layer must match the inputs of the following layer
--i.e. the number of nodes in layer n must match the number of inputs of nodes in layer n+1
type NodeNetwork a = [NodeLayer a]

randomList :: (RandomGen t) => t -> Int -> ([Int],t)
randomList gen len = (map (\x -> rem x 10) . take len $ randoms gen, gen)

nodeGen :: (RandomGen t) => Int -> t -> (Node Float, t)
nodeGen width gen =
    let (wInts,newGen) = randomList gen width
        weights = map fromIntegral wInts
    in (weights, newGen)

--generates a random layer in a network
layerGen :: (RandomGen t) => Int -> Int -> t -> NodeLayer Float
layerGen _ 0 _ = []
layerGen width len gen =
    let (oldGen, midGen) = split gen
        (newNode, newGen) = nodeGen width midGen
    in newNode : (layerGen width (len - 1) newGen)

--generates a random network
netGen :: (RandomGen t) => Int -> Int -> Int -> Int -> t -> NodeNetwork Float
netGen width 0 _ outSize gen = [layerGen width outSize gen]
netGen width length 0 outSize gen = 
    let (oldGen,newGen) = split gen
    in (layerGen width width oldGen): (netGen width (length - 1) 0 outSize newGen)
netGen width length inSize outSize gen = 
    let (oldGen,newGen) = split gen
    in (layerGen inSize width oldGen): (netGen width (length - 1) 0 outSize newGen)

evalPartialNodeNetwork :: (Floating a, Show a) => NodeNetwork a -> Int -> Inputs a -> Outputs a
evalPartialNodeNetwork network n inputs =
    evalNodeNetwork (take n network) inputs

--sigmoid is the activation function
--Basically, maps (-inf,inf) to (0,1) and is easily differentiable
sigmoid :: (Floating a, Show a) => a -> a
sigmoid z = 1 / (1 + exp (-z))

--derivative of sigmoid
sigmoid' :: (Floating a, Show a) => a -> a
sigmoid' z = (sigmoid z) * (1 - (sigmoid z))


--spits out a slightly better network (hopefully)
trainNetwork :: (Floating a, Show a) => NodeNetwork a -> (Inputs a -> Outputs a) -> [Inputs a] -> a -> NodeNetwork a
trainNetwork network ideal_func inputses eta =
    let idealses = map ideal_func inputses
        histories =  {-# SCC "histories" #-} map (keepEvalNodeNetwork network) inputses
        delses =  {-# SCC "delses" #-} getZipList $ delNetwork network <$> ZipList histories <*> ZipList idealses
        delWeightses =  {-# SCC "delWeightses" #-} getZipList $ applyDelNetwork <$> ZipList histories <*> ZipList delses

        avg :: (Floating a, Show a) => [a] -> a
        avg xs = (sum xs) / (fromIntegral (length xs))
        --jesus fuck
        weightAverager :: (Floating a, Show a) => [[[[a]]]] -> [[[a]]]
        weightAverager ns = Data.List.transpose $ map (map (map avg)) $ map (map Data.List.transpose) $ Data.List.transpose $ map Data.List.transpose $ Data.List.transpose $ ns
        delWeightAvg =  {-# SCC "weightAverager" #-} weightAverager delWeightses
    in applyDelWeights network delWeightAvg eta

applyDelWeights :: (Floating a, Show a) => NodeNetwork a -> [[[a]]] -> a -> NodeNetwork a
applyDelWeights network delWeights eta = 
    let applyNode n w = 
            let f w dw = w - (eta * (dw))
            in getZipList $ f <$> ZipList n <*> ZipList w
        applyLayer l ws = getZipList $ applyNode <$> ZipList l <*> ZipList ws
    in getZipList $ applyLayer <$> ZipList network <*> ZipList delWeights

tracer :: (Show a) => a -> a
tracer a | Debug.Trace.trace (show a) False = undefined
tracer a = a
 
    
--Just the cross product between weights and inputs
crossInputs :: (Floating a, Show a) => Weights a -> Inputs a -> a
crossInputs ws is = sum $ getZipList $ (*) <$> ZipList ws <*> ZipList is

--Gets the final sigmoid value of a node with a given input
evalNode :: (Floating a, Show a) => Node a -> Inputs a -> a
evalNode weights inputs = 
    sigmoid $ defaultBias + (crossInputs weights inputs)

--evaluates an entire layer with one input
--input size must match the input size of the nodes
evalNodeLayer :: (Floating a, Show a) => NodeLayer a -> Inputs a -> Outputs a
--evalNodeLayer layer inputs = evalNodeLayer' layer inputs `seq` tracer $ map (flip evalNode $ inputs) layer
evalNodeLayer layer inputs =
    let inMatrix = fromList (length inputs) 1 inputs
        layerMatrix = fromLists (layer)
    in {-# SCC "evalNodeLayer" #-} map sigmoid . concat . toLists $ multStd2 layerMatrix inMatrix

--given the output of all previous layers, it adds it's own output to the end
keepEvalNodeLayer :: (Floating a, Show a) => NodeLayer a -> [Outputs a] -> [Outputs a]
keepEvalNodeLayer layer history =
    let input = last history
        output = evalNodeLayer layer input
    in history ++ [output]

--returns a list of the output of each node
keepEvalNodeNetwork :: (Floating a, Show a) => NodeNetwork a -> Inputs a -> [Outputs a]
keepEvalNodeNetwork network inputs = foldl' (flip keepEvalNodeLayer) [inputs] network

--gets the delta thing for the last layer
delLayerLast :: (Floating a, Show a) => Outputs a -> Outputs a -> Inputs a
delLayerLast output ideal = 
        getZipList $ (\x t -> (x - t) * x * (1 - x)) <$> ZipList output <*> ZipList ideal

--prev layer is actually the next layer
--It's just prev because we're going backwards
delLayer :: (Floating a, Show a) => NodeLayer a -> NodeLayer a -> [a] -> Outputs a -> [a]
delLayer layer prevLayer prevDel output =
    let applyD d ws = map (*d) ws
        dSum = map sum $ Data.List.transpose $ getZipList $ applyD <$> ZipList prevDel <*> ZipList prevLayer
    in getZipList $ (\d o -> d * o * (1 - o)) <$> ZipList dSum <*> ZipList output
        
--gets the delta things for each node in a network via backpropogation
delNetwork :: (Floating a, Show a) => NodeNetwork a -> [Outputs a] -> Outputs a -> [[a]]
delNetwork network history ideal =
    let lastDelLayer = delLayerLast (last history) ideal
        lastLayer = last network
        --secondLastOutput = last . init $ history
        --secondLastLayer = last . init $ network
        histAndLayer = reverse $ init $ zip (tail history) network
        --a foldable version of delLayer
        f (pds, pl) (o, l) = 
            let d = delLayer l pl (head pds) o
            in (d:pds, l)
        in (\(d,_) -> d) $ foldl' f ([lastDelLayer], lastLayer) histAndLayer

--gets the things to multiply each weight by
applyDelNetwork :: (Floating a, Show a) => [Outputs a] -> [[a]] -> [[[a]]]
applyDelNetwork history del =
    let f d o = map (\x -> map (*x) o) d
    in getZipList $ f <$> ZipList (del) <*> ZipList (init (history))

--Evaluates a layer, and then feeds the outputs into the next layer's inputs
evalNodeNetwork :: (Floating a, Show a) => NodeNetwork a -> Inputs a -> Outputs a
evalNodeNetwork network inputs = foldl' (flip evalNodeLayer) inputs network

cost :: (Floating a, Show a) => NodeNetwork a -> (Inputs a -> Outputs a) -> [Inputs a] -> a
cost network ideal inputs_vector = 
    let ideal_out = map ideal inputs_vector
        net_out = map (evalNodeNetwork network) inputs_vector
        num_inputs = length inputs_vector
        magSq (x:xs) = x * x + (magSq xs)
        magSq [] = 0
        squareError y a = magSq . getZipList $ (-) <$> ZipList y <*> ZipList a 
    in (*) (0.5 / (fromIntegral num_inputs)) . sum . getZipList $ squareError <$> ZipList ideal_out <*> ZipList net_out
