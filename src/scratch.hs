import Data.List

avg xs = (sum xs) / (fromIntegral (length xs))

f xs = map (map (map avg)) $ map (map transpose) $ transpose $ map transpose $ transpose xs
