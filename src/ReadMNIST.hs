{-#LANGUAGE BangPatterns #-}
module ReadMNIST(getLabels, getImages) where

import qualified Data.ByteString.Lazy as BL
import Data.Binary.Get
import Data.Word
import Data.List.Split

--reads a labels files and returns the list of labels
getLabels :: FilePath -> IO (Integer, Integer, [Integer])
getLabels filename = do
    !contents <- BL.readFile filename
    let (m,a,l) = runGet labelParse contents
    return (toInteger m, toInteger a, map toInteger l)

--reads an images file and gets the images
getImages :: FilePath -> IO (Integer, Integer, [[Integer]])
getImages filename = do
    !contents <- BL.readFile filename
    let (_, _, rows, cols, pixels) = runGet imageParse contents
    return (toInteger rows, toInteger cols, chunksOf (fromIntegral . toInteger $ rows * cols) (map toInteger pixels))
    

--gets the header and the labels from a bytestring
labelParse :: Get (Word32, Word32, [Word8])
labelParse = do
    !magic <- getWord32be
    !amount <- getWord32be
    !labels <- listOfLabels
    return (magic, amount, labels)

--gets the labels from a bytestring
listOfLabels :: Get [Word8]
listOfLabels = do
    empty <- isEmpty
    if empty 
        then return []
        else do
            l <- getWord8
            rest <- listOfLabels
            return (l:rest)

--gets the header and the pixels from a image bytesting
imageParse :: Get (Word32, Word32, Word32, Word32, [Word8])
imageParse = do
    !magic <- getWord32be
    !amount <- getWord32be
    !rows <- getWord32be
    !cols <- getWord32be
    !pixels <- listOfPixels
    return (magic, amount, rows, cols, pixels)

--gets a list of pixels from bytestring
listOfPixels :: Get [Word8]
listOfPixels = do
    empty <- isEmpty
    if empty
        then return []
        else do
            p <- getWord8
            rest <- listOfPixels
            return (p:rest)
