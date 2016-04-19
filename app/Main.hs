{-#LANGUAGE BangPatterns #-}
import NodeNetwork
import ReadMNIST
import System.Random
import Data.Maybe
import Control.DeepSeq

main = main'

--the actual handwriting data is loaded
main' = do
    let labelPath = "labels.bin"
    let imagePath = "images.bin"
    (m, a, ls) <- getLabels labelPath
    (r, c, ps) <- getImages imagePath

    gen <- getStdGen
    let net = netGen 2 2 (fromIntegral (r * c)) 4 $ gen
    --remember: the image is the input
    let io = ideal_out ps ls
    let iof = map (map fromIntegral) ps :: [[Float]]
    
    printTrain (-1) net io ps 3
    --print $ map (evalNodeNetwork net) iof
    
--a test using xor as the training function
test = do
    gen <- getStdGen
    let net = netGen 5 5 2 4 gen
    let inputs =  [[0,0],[0,1],[1,0],[1,1]]
    let inputsf = map (map fromIntegral) inputs :: [[Float]]
    
    net' <- printTrain 100000 net xor' inputs 5
    print "old:"
    print $ map (evalNodeNetwork net) inputsf
    print "new:"
    print $ map (evalNodeNetwork net') inputsf


printTrain n net io ps size =
    if n == 0 
        then return net
        else do
            s <- sample ps size
            --let sn = show net
            let c = cost net io s
            --print net
            print c
            let net' = trainNetwork net io s 0.1
            printTrain (n-1) net' io ps size

xor' [0,0] = [1,0,0,0]
xor' [0,1] = [0,1,0,0]
xor' [1,0] = [0,0,1,0]
xor' [1,1] = [0,0,0,1]

ideal_out :: [[Integer]] -> [Integer] -> [Float] -> [Float]
ideal_out ps ls input = 
    let pairings = (zip (map (map fromIntegral) ps) ls) :: [([Float], Integer)]
    in digitToBinary . fromJust . lookup input $ pairings

sample :: [[Integer]] -> Int -> IO [[Float]]
sample xs 0 = return []
sample xs n = do
    p <- pick xs
    let pf = (map fromIntegral p) :: [Float]
    ps <- sample xs (n-1)
    return (pf:ps)

pick :: [a] -> IO a
pick xs = randomRIO (0, length xs - 1) >>= return . (xs !!)

digitToBinary :: (Num a, Eq a) => a -> [Float]
digitToBinary 0 = [0,0,0,0]
digitToBinary 1 = [0,0,0,1]
digitToBinary 2 = [0,0,1,0]
digitToBinary 3 = [0,0,1,1]
digitToBinary 4 = [0,1,0,0]
digitToBinary 5 = [0,1,0,1]
digitToBinary 6 = [0,1,1,0]
digitToBinary 7 = [0,1,1,1]
digitToBinary 8 = [1,0,0,0]
digitToBinary 9 = [1,0,0,1]

