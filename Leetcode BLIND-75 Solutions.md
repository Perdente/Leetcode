
<details>
<summary>Two Sum</summary>
<ul>
    Given an array of integers nums and an integer $target$, return indices of the two numbers such that they add up to $target$.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

 

Example 1:

Input: nums = $[2,7,11,15]$, target = $9$
Output: $[0,1]$
Explanation: Because $nums[0] + nums[1] == 9$, we return $[0, 1]$.
<details>
<summary>Approach</summary>
<ul>
Here, we always need to see $target - nums[i]$ if already interacted with us or not. if yes then answer exists.  
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
vector<int> twoSum(vector<int>& nums, int target) {
    map<int, int> mp;
    vector<int> ans;
    for (int i = 0; i < (int)nums.size(); ++i) {
        if (mp.count(target - nums[i])) {
            ans.push_back(i), ans.push_back(mp[target - nums[i]]); break;
        }
        mp[nums[i]] = i;
    }
    return ans;
}
```

</ul>
</details>

</ul>
</details>

 
<details>
<summary>Best Time to Buy and Sell Stock</summary>
<ul>
    You are given an array prices where $prices[i]$ is the price of a given stock on the $i$th day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return $0$.

 

Example 1:

Input: prices = $[7,1,5,3,6,4]$
Output: $5$
Explanation: Buy on day $2$ (price = $1$) and sell on day $5$ (price = $6$), profit = $6-1 = 5$.
Note that buying on day $2$ and selling on day $1$ is not allowed because you must buy before you sell.
<details>
<summary>Approach</summary>
<ul>
Here, our left pointer must be lowest possible and right pointer is highest possible. So, we initialize left to be $0$ and right to be $1$. When we find $nums[left] > nums[right]$ we update our left pointer with right. otherwise we calculate total distance.
Final answer is the maximum total distance of left and right.
    
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
int maxProfit(vector<int>& prices) {
    int left = 0, right = 1, ma = 0;
    while (right < (int)prices.size()) {
        if (prices[left] > prices[right]) {
            left = right;
        } else {
            ma = max(ma, prices[right] - prices[left]);
        }
        right++;
    }
    return ma;
}
```

</ul>
</details>

</ul>
</details>


<details>
<summary>Contains Duplicate</summary>
<ul>
    Given an integer array $nums$, return $true$ if any value appears at least twice in the array, and return $false$ if every element is distinct.
Example 1:

Input: nums = $[1,2,3,1]$
Output: $true$
<details>
<summary>Approach</summary>
<ul>
Similar to Two Sum problem. insert value in set and check if it's already present or not.
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
bool containsDuplicate(vector<int>& nums) {
    set<int> st;
    for (auto val: nums) {
        if (st.count(val)) return true;
        st.insert(val);
    }
    return false;
}
```

</ul>
</details>

</ul>
</details>
    

<details>
<summary>Product of Array Except Self</summary>
<ul>
Given an integer array nums, return an array answer such that $answer[i]$ is equal to the product of all the elements of nums except $nums[i]$.

The product of any prefix or suffix of nums is guaranteed to fit in a $32$-bit integer.

You must write an algorithm that runs in $O(n)$ time and without using the division operation.

Example 1:

Input: nums = $[1,2,3,4]$
Output: $[24,12,8,6]$    
<details>
<summary>Approach</summary>
<ul>
Here, for $i$ th index we need to calculate $prefix[i - 1] * sufix[i + 1]$.  
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++

vector<int> productExceptSelf(vector<int>& nums) {
    int temp = 1;
    int n = nums.size();
    vector<int> prefix(n), suffix(n);
    for (int i = 0; i < n; ++i) {
        temp *= nums[i];
        prefix[i] = temp;
    }
    temp = 1;
    for (int i = n - 1; i >= 0; --i) {
        temp *= nums[i];
        suffix[i] = temp;
    }
    vector<int> ans(n);
    for (int i = 0; i < n; ++i) {
        int tot = 1;
        if (i - 1 >= 0) tot *= prefix[i - 1];
        if (i + 1 < n) tot *= suffix[i + 1];
        ans[i] = tot;
    }
    return ans;
}
```

</ul>
</details>

</ul>
</details>


<details>
<summary>Maximum Subarray</summary>
<ul>
Given an integer array $nums$, find the contiguous subarray (containing at least one number) which has the largest $sum$ and return its $sum$.

A subarray is a contiguous part of an array.

Example 1:

Input: nums = $[-2,1,-3,4,-1,2,1,-5,4]$
Output: $6$
Explanation: $[4,-1,2,1]$ has the largest sum = $6$.    
<details>
<summary>Approach</summary>
<ul>
Kadane's algorithm. We keep adding the $value$ to $sum$ until $sum$ is $(-ve)$. Each iteration we store the maximum $sum$ to $ans$ variable. return $ans$.
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
int maxSubArray(vector<int>& nums) {
    int ans = INT_MIN, sum = 0;
    for (auto v: nums) {
        sum += v;
        ans = max(ans, sum);
        sum = max(0, sum);
    }
    return ans;
}
```

</ul>
</details>

</ul>
</details>
    

<details>
<summary>Maximum Product Subarray</summary>
<ul>
 Given an integer array $nums$, find a contiguous non-empty subarray within the array that has the $largest$ product, and return the product.

The test cases are generated so that the answer will fit in a $32$-bit integer.

A subarray is a contiguous subsequence of the array.

Input: nums = $[2,3,-2,4]$
Output: $6$
Explanation: $[2,3]$ has the largest product $6$.   
<details>
<summary>Approach</summary>
<ul>
If there is no $(-ve)$ value or even number of $(-ve)$ values then answer is just total array multiplied.Here, either I can choose $i$ th index as a continuous sub-array element or I can start my new array with $i$ th index.  But when a $(-ve)$ value occurs then our maximum multiplied value is minimized and vice versa. So, we have to calculate both maximum multiplication and minimum multiplication till $i$ and if $(-ve)$ value occurs then we can simply swap both variable and calculate maximum out of it.
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
int maxProduct(vector<int>& nums) {
    int ma = 0, mi = 1000;
    int ans = INT_MIN;
    for (auto val: nums) {
        if (val < 0) swap(ma, mi);
        ma = max(val, val * ma);
        mi = min(val, val * mi);
        ans = max(ans, ma);
    }
    return ans;
}
```

</ul>
</details>

</ul>
</details>

    
<details>
<summary>Find Minimum in Rotated Sorted Array</summary>
<ul>
    Suppose an array of length $n$ sorted in ascending order is rotated between $1$ and $n$ times. For example, the array nums = $[0,1,2,4,5,6,7]$ might become:

$[4,5,6,7,0,1,2]$ if it was rotated $4$ times.
$[0,1,2,4,5,6,7]$ if it was rotated $7$ times.
Notice that rotating an array $[a[0], a[1], a[2], ..., a[n-1]]$ $1$ time results in the array $[a[n-1], a[0], a[1], a[2], ..., a[n-2]]$.

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in $O(log n)$ time.

Example 1:

Input: nums = $[3,4,5,1,2]$
Output: $1$
Explanation: The original array was $[1,2,3,4,5]$ rotated $3$ times.
    
<details>
<summary>Approach</summary>
<ul>
    Here, we need to find the pivot point where there is a break. $(3, 4), (4, 5), (5, 1), (1, 2)$ here, only $(5, 1)$ point has decreasing tuple. And by observation we can see that the left portion of the pivot is always be greater and right is always smaller. So, we can run Binary Search on that Pivot point, 
    
```
if nums[mid] >= nums[left]
    search right portion
else 
    search left portion.
```
    
But if the array is rotated $n$ times then it's already sorted. Here, our $nums[0]$ is the answer. So, our final result would be $min(nums[0], nums[right])$.

</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
int findMin(vector<int>& nums) {
    int n = nums.size();
    int left = 0, right = n - 1;
    while (right > left + 1) {
        int mid = (left + right) / 2;
        if (nums[mid] >= nums[left]) {
            left = mid;
        } else {
            right = mid;
        }
    }

    return min(nums[0], nums[right]);
}
```

</ul>
</details>

</ul>
</details>
    
    
<details>
<summary>Search in Rotated Sorted Array</summary>
<ul>
There is an integer array $nums$ sorted in ascending order (with _distinct_ values).

Prior to being passed to your function, $nums$ is possibly rotated at an unknown pivot index $k (1 <= k < nums.length)$ such that the resulting array is $[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed)$

Given the array nums after the possible rotation and an integer $target$, return the index of target if it is in nums, or $-1$ if it is not in nums.

You must write an algorithm with $O(log n)$ runtime complexity.

 

Example 1:

Input: nums = $[4,5,6,7,0,1,2]$, target = $0$
Output: $4$   
<details>
<summary>Approach</summary>
<ul>
    Here, there is two portion of sorted array. We need to check in which part our $target$ value appears. If $nums[left] <= nums[mid]$ then we are in the left portion. Now if our $target$ value is inside this left portion we search in $left$ or vice versa.
                                                                                                                       
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
int search(vector<int>& nums, int target) {
    int n = nums.size();
    int left = 0, right = n - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (nums[mid] == target) return mid;
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target and target <= nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (nums[mid] <= target and target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return -1;
}
```

</ul>
</details>

</ul>
</details>

<details>
<summary>3Sum</summary>
<ul>
    Given an integer array $nums$, return all the triplets $[nums[i], nums[j], nums[k]]$ such that $i != j, i != k,$ and $j != k$ and $nums[i] + nums[j] + nums[k] = 0$.

Notice that the solution set must not contain duplicate triplets.

Example 1:

Input: nums = $[-1,0,1,2,-1,-4]$
Output: $[[-1,-1,2],[-1,0,1]]$
<details>
<summary>Approach</summary>
<ul>
Here, after sorting the array we first fix the first value and then use two pointer to get the next two values. As the triplets can't contain any duplicate so we increase our pointer when we have adjacent elements equal. Here, $j$ th index is the rightmost value and we don't need to check for adjacent here, as it's been processed in main $while$ loop
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        set<vector<int>> ans;
        int n = nums.size();
        for (int i = 0; i < n - 2; ++i) {
            int j = i + 1;
            int k = n - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == 0) {
                    vector<int> temp {nums[i], nums[j], nums[k]};
                    ans.insert(temp);
                    j++, k--;
                } else if (sum > 0) k--;
                else if (sum < 0) j++;
            }
        }
        vector<vector<int>> result(ans.begin(), ans.end());
        
        return result;
    }
};
```

</ul>
</details>

</ul>
</details>
    
    

<details>
<summary>Container With Most Water</summary>
<ul>
You are given an integer array height of length $n$. There are $n$ vertical lines drawn such that the two endpoints of the $i$th line are $(i, 0) and (i, height[i])$.

Find two lines that together with the $x-axis$ form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.
![alt text](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

Example 1:


Input: height = $[1,8,6,2,5,4,8,3,7]$
Output: $49$
Explanation: The above vertical lines are represented by array $[1,8,6,2,5,4,8,3,7]$. In this case, the max area of water (blue section) the container can contain is 49.
    
<details>
<summary>Approach</summary>
<ul>
Simple two pointer. Here, as we only care about maximum area. So, width is as large as possible is necessary. So, we use two pointer from $l = 0$ to $r = n - 1$ and calculate the area from it. If left index value is less or equals right index value then we increase left pointer or vice versa.
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int n = height.size();
        int ma = 0;
        int left = 0, right = n - 1;
        while (left < right) {
            int h = min(height[left], height[right]);
            int w = right - left;
            ma = max(ma, h * w);
            if (height[left] <= height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return ma;
    }
};
```

</ul>
</details>

</ul>
</details>
    

<details>
<summary>Number of 1 Bits</summary>
<ul>
    Write a function that takes an unsigned integer and returns the number of $'1'$ bits it has (also known as the Hamming weight).
<details>
<summary>Approach</summary>
<ul>

</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int cnt=0;
        while(n){
          cnt++;
          n&=(n-1);
        }
        return cnt;
    }
};
    
class Solution {
public:
    int hammingWeight(uint32_t n) {
        return __builtin_popcount(n);
    }
};
```

</ul>
</details>

</ul>
</details>
    
<details>
<summary>Counting Bits</summary>
<ul>
Given an integer $n$, return an array ans of length $n + 1$ such that for each $i (0 <= i <= n)$, $ans[i]$ is the number of $1's$ in the binary representation of $i$.

 
<details>
<summary>Approach</summary>
<ul>

</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> ans;
        for (int i = 0; i <= num; ++i) {
            ans.push_back(__builtin_popcount(i));
        }
        return ans;
    }
};
```

</ul>
</details>

</ul>
</details>



<details>
<summary>Missing Number / MEX</summary>
<ul>
    Given an array $nums$ containing $n$ distinct numbers in the range $[0, n]$, return the only number in the range that is missing from the array.
<details>
<summary>Approach</summary>
<ul>
Here, we use a $set$ to store the values. Whenever, we've a $MEX$ we increment our $MEX$ value. Total time complexity $O(n2)$. 
Using XOR, we can first xor with $[0, n]$ and then xor it with the whole array. Only one value is not gonna cancel out which is our answer. Complexity $O(n)$
Using total sum, we know that sum for $[0, n]$ is $n * (n + 1) / 2$. So, we can keep decrement our sum from the array value and final sum is returned.Complexity $O(n)$
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int mex = 0;
        set<int> st;
        for (auto val: nums) {
            st.insert(val);
            while (st.count(mex)) mex++;
        }
        return mex;
    }
};
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int x = 0;
        int n = nums.size();
        for (int i = 0; i <= n; ++i) x ^= i;
        for (auto v: nums) x ^= v;
        return x;
    }
};
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        int sum = n * (n + 1) / 2;
        for (auto val: nums) sum -= val;
        return sum;
    }
};                              
```

</ul>
</details>

</ul>
</details>
    

<details>
<summary>Reverse Bits</summary>
<ul>
    Reverse bits of a given $32$ bits unsigned integer.
    Input: $n = 00000010100101000001111010011100$
Output:    $964176192 (00111001011110000010100101000000)$
Explanation: The input binary string $00000010100101000001111010011100$ represents the unsigned integer $43261596$, so return $964176192$ which its binary representation is $00111001011110000010100101000000$.
<details>
<summary>Approach</summary>
<ul>

</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        int ans = 0;
        for (int mask = 31; mask >= 0; --mask) {
            if (n & (1 << mask)) ans += (1 << (31 - mask));
        }
        return ans;
    }
};
```

</ul>
</details>

</ul>
</details>
    
<details>
<summary>Climbing Stairs</summary>
<ul>
 You are climbing a staircase. It takes $n$ steps to reach the top.

Each time you can either climb $1$ or $2$ steps. In how many distinct ways can you climb to the top?

Example 1:

Input: $n = 2$
Output: $2$
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps   
<details>
<summary>Approach</summary>
<ul>
Here, each step $(step >= 2)$ depends on it's two previous values. So, dp state would be $dp[i] = dp[i-1] + dp[i - 2]$
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    int climbStairs(int n) {
        vector<int> dp(n + 1);
        dp[0] = 1, dp[1] = 1;
        for (int i = 2; i <= n; ++i) {
            if (i - 1 >= 0) dp[i] += dp[i - 1];
            if (i - 2 >= 0) dp[i] += dp[i - 2];
        }
        return dp[n];
    }
};
```

</ul>
</details>

</ul>
</details>

    


<details>
<summary>Coin Change</summary>
<ul>
    You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the $fewest$ number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return $-1$.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = $[1,2,5]$, amount = $11$
Output: $3$
Explanation: $11 = 5 + 5 + 1$
<details>
<summary>Approach</summary>
<ul>
Here, we only add $i$th coin if $amount - coin[i]$ exists. So, our dp state would be $dp[i] = min (dp[i], 1 + dp[sum - coin[i]])$
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
    const int oo = 1e9 + 5;
public:
    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size(), sum = amount;
        vector<int> dp(sum + 1, oo);
        dp[0] = 0;
        for (int i = 1; i <= sum; ++i) {
            for (auto j: coins) {
                if (i - j >= 0) dp[i] = min(dp[i], 1 + dp[i - j]);
            }
        }
        return dp[sum] == oo ? -1 : dp[sum];
    }
};
```

</ul>
</details>

</ul>
</details>
    
    
<details>
<summary>Longest Increasing Subsequence</summary>
<ul>
Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, $[3,6,2,7]$ is a subsequence of the array $[0,3,1,6,2,2,7]$  
<details>
<summary>Approach</summary>
<ul>
Here, dp solution is $O(n^2)$. Here, for each index value we calculate LIS. If $i$ th index is stictly greater then we can add the value to the answer. Finally answer is maximum for each index.
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> LIS(n, 1);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    LIS[i] = max(LIS[i], 1 + LIS[j]);
                }
            }
        }
        int ans = INT_MIN;
        for (auto v: LIS) ans = max(ans, v);
        return ans;
    }
};
```

</ul>
</details>

</ul>
</details>
    
<details>
<summary>Longest Common Subsequence</summary>
<ul>
Given two strings $text1$ and $text2$, return the length of their longest common subsequence. If there is no common subsequence, return $0$.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, $ace$ is a subsequence of $abcde$.
A common subsequence of two strings is a subsequence that is common to both strings.

 

Example 1:

Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is $ace$ and its length is $3$. 
    
<details>
<summary>Approach</summary>
<ul>
Here, we will use top down approach. If $i$ the and $j$ th character matched then we will find solution for $i + 1$ and $j + 1$ character. Otherwise, we will find the maximum for $i + 1$ character for string 1 and $j + 1$ character for string 2.
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
    int dp[1004][1004];
    int lcs(int i, int j, string &text1, string &text2) {
        int n = text1.size(), m = text2.size();
        if (i == n or j == m) return 0;
        if (dp[i][j] != -1) return dp[i][j];
        if (text1[i] == text2[j]) return lcs(i + 1, j + 1, text1, text2) + 1;
        int x = 0, y = 0;
        x = lcs(i + 1, j, text1, text2);
        y = lcs(i, j + 1, text1, text2);
        // cout << dp[i][j] << endl;
        return dp[i][j] = max(x, y);
    }
public:
    
    
    int longestCommonSubsequence(string text1, string text2) {
        memset(dp, -1, sizeof dp);
        return lcs(0, 0, text1, text2);
    }
};
```

</ul>
</details>

</ul>
</details>
    
<details>
<summary>Word Break</summary>
<ul>
Given a string $s$ and a dictionary of strings $wordDict$ , return $true$ if $s$ can be segmented into a space-separated sequence of one or more dictionary words.

> Note that the same word in the dictionary may be reused multiple times in the segmentation.

 

Example 1:

Input: s = "leetcode", wordDict = ["leet","code"]
    
Output: true
    
Explanation: Return true because "$leetcode$" can be segmented as "$leet$" "$code$".
<details>
<summary>Approach</summary>
<ul>
Here, we check from $s$ string if any string from $st$ is present or not ; starting from index $pos$. So, for each recursive call we start from $pos$ index and look for a substring of length $i - pos + 1$ each time and check if this substring is present in our $st$. If present we search for $i + 1$ th index and if no substring matches then we return $false$.
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
    
public:
    // const int N = 1e3 + 4;
    int dp[1004];
    bool rec(int pos, string &s, set<string> &st) {
        if (pos >= (int)s.size()) return true;
        if (dp[pos] != -1) return dp[pos];

        for (int i = pos; i < (int)s.size(); ++i) {
            string temp = s.substr(pos, i - pos + 1);
            if (st.count(temp)) {
                if (rec(i + 1, s, st)) {
                    cout << temp << " " << pos << '\n';
                    return dp[pos] = true;
                }
            }
        }
        return dp[pos] = false;
    }

    bool wordBreak(string s, vector<string>& wordDict) {    
        set<string> st;
        for (auto it: wordDict) st.insert(it);
        memset(dp, -1, sizeof dp);
        return rec(0, s, st);
    }
};
```

</ul>
</details>

</ul>
</details>

   


<details>
<summary>Combination Sum</summary>
<ul>
Given an array of distinct integers candidates and a target integer $target$, return a list of all unique combinations of candidates where the chosen numbers sum to $target$. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

 

Example 1:

Input: candidates = $[2,3,6,7]$, target = $7$
Output: $[[2,2,3],[7]]$
Explanation:
$2$ and $3$ are candidates, and $2 + 2 + 3 = 7$. Note that $2$ can be used multiple times.
$7$ is a candidate, and $7 = 7$.
These are the only two combinations.
<details>
<summary>Approach</summary>
<ul>
Similar to Coin Combination II.

</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    vector<vector<int>> vec;
    vector<int> temp;
    
    void rec(int i, vector<int> arr, int sum) {
        if (sum == 0) {
            vec.push_back(temp);
            return;
        }
        if (i == (int)arr.size() or sum < 0) return;
        rec(i + 1, arr, sum);
        temp.push_back(arr[i]);
        rec(i, arr, sum - arr[i]);
        temp.pop_back();
    }
    
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        rec(0, candidates, target);
        return vec;
    }
};
```

</ul>
</details>

</ul>
</details>
    
    
<details>
<summary>House Robber</summary>
<ul>
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that $adjacent$ houses have security systems connected and it will automatically contact the police if $two$ adjacent houses were broken into on the same night.

Given an integer array $nums$ representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = $[1,2,3,1]$
Output: $4$
Explanation: Rob house $1$ (money = $1$) and then rob house $3$ (money = $3$).
Total amount you can rob = $1 + 3 = 4$.    
<details>
<summary>Approach</summary>
<ul>
Here, if we take values fron $i$ th house we have to take from $i + 2$ th house, we can't take from $i + 1$th house. So, our recurrence relation would be $$dp[i] = max(dp[i-1],nums[i] + dp[i - 2])$$.
</ul>
</details>

<details>
<summary>Code_iterative</summary>
<ul>
    
```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        int n=nums.size();
        if(n==1)return nums[0];
        vector<int> dp(n);
        dp[0]=nums[0];
        dp[1]=max(nums[0],nums[1]);
        for(int i=2;i<n;++i){
            dp[i]=max(dp[i-1],dp[i-2]+nums[i]);
        }
        
        return dp[n-1];
    }
};
```

</ul>
</details>

 <details>
<summary>Code_recursive</summary>
<ul>
    
```c++
class Solution {
public:
    int dp[406];
    int rec(int i, vector<int> v) {
        int n = v.size();
        if (i >= n) return 0;
        if (dp[i] != -1) return dp[i];
        int x = rec(i + 1, v);
        int y = v[i] + rec(i + 2, v);
        return dp[i] = max(x, y);
    }
    
    int rob(vector<int>& nums) {
        memset(dp, -1, sizeof dp);
        return rec(0, nums);
    }
};
```

</ul>
</details>

</ul>
</details>
    
    
<details>
<summary>House Robber II</summary>
<ul>
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

Example 1:

Input: nums = $[2,3,2]$
Output: $3$
Explanation: You cannot rob house $1$ (money = $2$) and then rob house $3$ (money = $2$), because they are adjacent houses.   
<details>
<summary>Approach</summary>
<ul>
The main difference between H_R1 and H_R2 is that here we can't calculate dp values for $nums[0]$ and $nums[n-1]$ together as they are adjacent. But we can use two dp arrays $1$st $[0,n-2]$ and $2$nd $[1,n-1]$ skipping both first and last values.Final answer is $max(dp[n - 1], dp[n - 2])$
</ul>
</details>

<details>
<summary>Code_iterative</summary>
<ul>
    
```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        int n=nums.size();
        if(n==1)return nums[0];
        if(n==2)return max(nums[0],nums[1]);
        vector<int> dp(n),dp1(n);
        dp[0]=nums[0];
        dp[1]=max(nums[0],nums[1]);
        for(int i=2;i<n-1;++i){
            dp[i]=max(dp[i-1],dp[i-2]+nums[i]);
        }
        dp1[1]=nums[1];
        dp1[2]=max(nums[1],nums[2]);
        for(int i=3;i<n;++i){
            dp1[i]=max(dp1[i-1],dp1[i-2]+nums[i]);
        }
        return max(dp[n-2],dp1[n-1]);
    }
};
```

</ul>
</details>


<details>
<summary>Code_recursive</summary>
<ul>
    
```c++
class Solution {
public:
    int dp[1005];
    
    int rec(int i, int end, vector<int> v) {
        int n = v.size();
        if (i > end) return 0;
        if (dp[i] != -1) return dp[i];
        int x = rec(i + 1, end, v);
        int y = v[i] + rec(i + 2, end, v);
        return dp[i] = max(x, y);
    }
    
    int rob(vector<int>& nums) {
        memset(dp, -1, sizeof dp);
        int n = nums.size();
        if (n == 1) return nums[0];
        int x = rec(0, n - 2, nums);
        memset(dp, -1, sizeof dp);
        int y = rec(1, n - 1, nums);
        return max(x, y);
        
    }
};
```

</ul>
</details>
    
</ul>
</details>
    
    
    
<details>
<summary>Decode Ways</summary>
<ul>
 A message containing letters from $A-Z$ can be encoded into numbers using the following mapping:

```
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
```

To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "$11106$" can be mapped into:

"$AAJF$" with the grouping $(1 1 10 6)$
"$KJF$" with the grouping $(11 10 6)$
Note that the grouping $(1 11 06)$ is invalid because "$06$" cannot be mapped into '$F$' since "$6$" is different from "$06$".

Given a string s containing only digits, return the number of ways to decode it.

The test cases are generated so that the answer fits in a $32$-bit integer.

 

Example 1:

Input: s = "$12$"
Output: $2$
Explanation: "$12$" could be decoded as "$AB$" $(1 2)$ or "$L$" $(12)$.   

<details>
<summary>Approach</summary>
<ul>

Say our string is $2126$, and we are at $1st$ position. So, can we include our next character $2$ in our answer? Well, we can if the next digit is $1 <= digit <= 9$. If the digit is $0$ then it can't contribute to the answer. Again, can we include next two digits? Yes until $26$ we have valid mapping. So, we check for next two digit if it's $10 <= two_digit <= 26$. If we reach the end of the string, then we've successfully completed one valid string, so $return 1$

</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    int dp[105];
    int rec(int i, string s) {
        int n = s.size();
        if (i >= n) return 1;
        if (dp[i] != -1) return dp[i];
        
        int ways = 0;
        
        int one_digit = s[i] - '0';
        if (1 <= one_digit and one_digit <= 9) ways += rec(i + 1, s);
        
        if (i + 1 < n) {
            int two_digit = (s[i] - '0') * 10 + (s[i + 1] - '0');
            if (10 <= two_digit and two_digit <= 26) ways += rec(i + 2, s);
        }
        return dp[i] = ways;
    }
    
    int numDecodings(string s) {
        memset(dp, -1, sizeof dp);
        return rec(0, s);
    }
};
```

</ul>
</details>

</ul>
</details>

    
<details>
<summary>Unique Paths</summary>
<ul>
There is a robot on an $m x n$ grid. The robot is initially located at the top-left corner (i.e., $grid[0][0]$). The robot tries to move to the bottom-right corner (i.e., $grid[m - 1][n - 1]$). The robot can only move either down or right at any point in time.

Given the two integers $m$ and $n$, return the number of possible unique paths that the robot can take to reach the bottom-right corner.    
<details>
<summary>Approach</summary>
<ul>
    
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        int dp[105][105];
        memset(dp, 0, sizeof dp);
        for (int i = 1; i <= n; ++i) dp[i][1] = 1;
        
        for (int i = 1; i <= m; ++i) dp[1][i] = 1;
        
        for (int i = 2; i <= n; ++i) {
            for (int j = 2; j <= m; ++j) {
                dp[i][j] += dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[n][m];
    }
};
```

</ul>
</details>

</ul>
</details>
    
    
<details>
<summary>Jump Game</summary>
<ul>
You are given an integer array $nums$. You are initially positioned at the array's $1st$ index, and each element in the array represents your maximum jump length at that position.

Return $true$ if you can reach the $last$ index, or $false$ otherwise.

 

Example 1:

Input: nums = $[2,3,1,1,4]$
Output: $true$
Explanation: Jump $1$ step from index $0$ to $1$, then $3$ steps to the last index.
    
<details>
<summary>Approach</summary>
<ul>
Here, from each particular position we calculate how far we can go. Next, we try for that farthest point how much we can go from there. If we reach greater or equal $n - 1$ th position we can reach the end. But if our second value is less than first value answer is false; 
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    
    bool canJump(vector<int>& nums) {
        int n = nums.size();
        int jump = 0;
        pair<int, int> interval = {0, 0};
        while (true) {
            jump++;
            int maxReach = -1;
            for (int i = interval.first; i <= interval.second; ++i) {
                maxReach = max(maxReach, i + nums[i]);
            }
            if (maxReach >= n - 1) {
                cout << jump << '\n';
                return true;
            }
            interval = {interval.second + 1, maxReach};
            if (interval.first > interval.second) return false;
        }
    }
};
```

</ul>
</details>

</ul>
</details>
  
<details>
<summary> Course Schedule</summary>
<ul>
There are a total of numCourses courses you have to take, labeled from $0$ to $numCourses - 1$. You are given an array prerequisites where $prerequisites[i] = [a_i, b_i]$ indicates that you must take course $b_i$ first if you want to take course $a_i$.

For example, the pair $[0, 1]$, indicates that to take course $0$ you have to first take course $1$.
Return $true$ if you can finish all courses. Otherwise, return $false$.
<details>
<summary>Approach</summary>
<ul>
Prerequisite: Cycle detection in a directed graph
Here, unlike undirected graph we use two $visited$ array. If one of the nodes work is ended then we before going to backtrack make our first $visited[node] = false$ as we can visit this same node via different path. In some case if both the $visted$ array value is $true$ then we are sure that there is a cycle. In this problem, if cycle is detected then return $false$ or vice-versa.  
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    vector<int> g[100005];
    bool vis[100005], dfsVis[100005];
    
    bool dfs(int u) {
        vis[u] = true;
        dfsVis[u] = true;
        for (auto v: g[u]) {
            if (!vis[v]) {
                if (dfs(v)) return true;
            } else if (dfsVis[v]) return true;
        }
        dfsVis[u] = false;
        return false;
    }
    
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        memset(vis, false, sizeof vis);
        memset(dfsVis, false, sizeof dfsVis);
        for (int i = 0; i < (int) prerequisites.size(); ++i) {
            int u = prerequisites[i][0], v = prerequisites[i][1];
            g[u].push_back(v);
        }
        for (int i = 0; i < numCourses; ++i) {
            if (!vis[i]) {
                if (dfs(i)) return false;
            }    
        }
        return true;
    }
    
};
```

</ul>
</details>

</ul>
</details>
     
    
<details>
<summary>Number of Islands</summary>
<ul>
    Given an $m x n$ $2$D binary grid grid which represents a map of $'1'$s (land) and $'0'$s (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
``` 

<details>
<summary>Approach</summary>
<ul>

</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        if(grid.empty() or grid[0].empty()) return 0;
        
        int H=grid.size();
        int W=grid[0].size();
        int ans=0;
        auto inside =[&](int row,int col){
            return 0<=row and row<H and 0<=col and col<W;
        };
        
        vector<pair<int,int>>directions{{1,0},{0,1},{-1,0},{0,-1}};
        vector<vector<bool>>vis(H,vector<bool>(W));
        for(int row=0;row<H;++row){
            for(int col=0;col<W;++col){
                if(!vis[row][col] and grid[row][col]=='1')
                {
                    ans++;
                    vis[row][col]=true;
                    queue<pair<int,int>>q;
                    q.push({row,col});
                    while(!q.empty()){
                        pair<int,int>p=q.front();
                        q.pop();
                        for(pair<int,int>dir:directions){
                            int new_row=p.first+dir.first;
                            int new_col=p.second+dir.second;
                            
                            if(inside(new_row,new_col) and !vis[new_row][new_col] and grid[new_row][new_col]=='1')
                            {
                                vis[new_row][new_col]=true;
                                q.push({new_row,new_col});
                            }
                                
                        }
                    }
                }
            }
        }
        return ans;
    }
};
```

</ul>
</details>

</ul>
</details>
    
    
   
<details>
<summary>Longest Consecutive Sequence</summary>
<ul>
Given an unsorted array of integers $nums$, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in $O(n)$ time.

Example 1:

Input: nums = $[100,4,200,1,3,2]$
Output: $4$
Explanation: The longest consecutive elements sequence is $[1, 2, 3, 4]$. Therefore its length is $4$. 
<details>
<summary>Approach</summary>
<ul>
$[100,4,200,1,3,2] => [100] [200] [1, 2, 3, 4]$  Here, we need to figure out the start of the segment which is can be done by checking $st.count(val - 1)$ exists or not? Then, we count the sequence length and print the maximum of 'em.
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> st(nums.begin(), nums.end());
        int ma = 0;
        for (auto it: nums) {
            int cnt = 0;
            if (!st.count(it - 1)) {
                int temp = it;
                while (st.count(temp)) cnt++, temp++;
            }
            ma = max(ma, cnt);
        }
        return ma;
    }
};
```

</ul>
</details>

</ul>
</details>
 
<details>
<summary>Alien Dictionary </summary>
<ul>
Given a sorted dictionary of an alien language having $N$ words and $k$ starting alphabets of standard dictionary. Find the order of characters in the alien language.
https://practice.geeksforgeeks.org/problems/alien-dictionary/1#
    
```
Input: 
N = 5, K = 4
dict = {"baa","abcd","abca","cab","cad"}
Output:
1
Explanation:
Here order of characters is 
'b', 'd', 'a', 'c' Note that words are sorted 
and in the given language "baa" comes before 
"abcd", therefore 'b' is before 'a' in output.
Similarly we can find other orders.
```
<details>
<summary>Approach</summary>
<ul>
Prerequisite : Topological sorting. Here, from two adjacent element from the string array the character that came before is lexicographically smaller. So, we can have a directed graph whose parent node is bigger and child node is lexicographically smaller according to alien dictionary.

 $dict = {"baa","abcd","abca","cab","cad"}$ here, the directed graph looks like
    ```
    1. b -> a -> c
    2. b -> d -> a
    ```
So, we can use dfs traversal and while backtracking we push back the last visited character into a string. Finally, we have our final string as decending order. We need to reverse the string to get our answer :) 
    
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution{
    public:
    
    void dfs(int u, vector<vector<int>> &graph, vector<bool> &vis, string &ans) {
        vis[u] = true;
        for (auto v: graph[u]) {
            if (!vis[v]) dfs(v, graph, vis, ans);
        }
        ans += (char)u + 'a'; 
    }
    string findOrder(string dict[], int N, int K) {
        vector<vector<int>> graph(K);
        for (int i = 0; i < (int) N - 1; ++i) {
            string a = dict[i], b = dict[i + 1];
            int n = min((int) a.size(), (int) b.size());
            for (int ch = 0; ch < n; ++ch) {
                if (a[ch] != b[ch]) {
                    graph[a[ch] - 'a'].push_back(b[ch] - 'a');
                    break;
                }
            }
        }
        vector<bool> vis(K);
        string ans;
        for (int i = 0; i < K; ++i) {
            if (!vis[i]) {
                dfs(i, graph, vis, ans);
            }
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
};
```

</ul>
</details>

</ul>
</details>
 
    
    
<details>
<summary>Graph Valid Tree</summary>
<ul>
Given $n$ nodes labeled from $0$ to $n - 1$ and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree. 

```
Input: n = 5 edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
Output: false.
```

<details>
<summary>Approach</summary>
<ul>

> Conditions for a graph to be a tree - 
- Undirected graph.
- Always single component graph.
- Can't have any cycles.

> Conditions for a graph having cycle -
- A graph with $n$ vertices is a tree if and only if it has $n - 1$ edges.
- While traversing a graph if a node is visited twice and it's not a parent node, then the graph must contains a cycle.
    

</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:

    bool dfs(int u, int parent, vector<bool> &vis, vector<vector<int>> &g) {
        vis[u] = true;
        for (auto v: g[u]) {
            if (!vis[v]) {
                if (dfs(v, u, vis, g)) return true;
            }
            else if (v != parent) return true; 
        }
        return false;
    } 

    bool validTree(int n, vector<vector<int>> &edges) {
        vector<bool> vis(n);
        int m = edges.size();
        vector<vector<int>> g(n);
        for (int i = 0; i < m; ++i) {
            g[edges[i][0]].push_back(edges[i][1]);
            g[edges[i][1]].push_back(edges[i][0]);
        }
        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            if (!vis[i]) {
                cnt++;
                if (dfs(i, -1, vis, g)) {
                    return false; 
                }
            } 
        }
        return (cnt == 1);
    }
};
```

</ul>
</details>

</ul>
</details>
    

<details>
<summary>Merge Intervals</summary>
<ul>
Given an array of intervals where $intervals[i] = [start_i, end_i]$, merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

 

Example 1:

Input: intervals = $[[1,3],[2,6],[8,10],[15,18]]$
Output: $[[1,6],[8,10],[15,18]]$
Explanation: Since intervals $[1,3]$ and $[2,6]$ overlaps, merge them into $[1,6]$.   
<details>
<summary>Approach</summary>
<ul>
In diary...
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
// https://www.youtube.com/watch?v=_FkR5zMwHQ0
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> ans;
        sort(intervals.begin(), intervals.end(), [](auto x, auto y){return x[0] < y[0];});
        for (auto interval: intervals) {
            if (ans.empty()) {
                ans.push_back(interval);
            } else {
                vector<int> prev = ans.back();
                if (interval[0] <= prev[1]) {
                    ans.back()[1] = max(prev[1], interval[1]);
                } else {
                    ans.push_back(interval);
                }
            }
        }
        return ans;
    }
};
```

</ul>
</details>

</ul>
</details>
    
    

<details>
<summary>Insert Interval</summary>
<ul>
You are given an array of non-overlapping intervals intervals where $intervals[i] = [start_i, end_i]$ represent the start and the end of the $i$th interval and intervals is sorted in ascending order by $start_i$. You are also given an interval $newInterval = [start, end]$ that represents the start and end of another interval.

Insert $newInterval$ into intervals such that $intervals$ is still sorted in ascending order by $start_i$ and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.
 

Example 1:

Input: intervals = $[[1,3],[6,9]]$, newInterval = $[2,5]$
Output: $[[1,5],[6,9]]$   
<details>
<summary>Approach</summary>
<ul>
In diary :)
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        vector<vector<int>> ans;
        int n = intervals.size(), i = 0;
        while (i < n and intervals[i][0] <= newInterval[0]) {
            ans.push_back(intervals[i++]);
        }
        if (ans.empty() or newInterval[0] > ans.back()[1]) ans.push_back(newInterval);
        else {
            ans.back()[1] = max(ans.back()[1], newInterval[1]);
        }
        while (i < n) {
            if (intervals[i][0] <= ans.back()[1]) {
                ans.back()[1] = max(ans.back()[1], intervals[i][1]);
            } else {
                ans.push_back(intervals[i]);
            }
            i++;
        }
        return ans;
    }
};
```

</ul>
</details>

</ul>
</details>
    
    
    
    
<details>
<summary>Non-overlapping Intervals</summary>
<ul>
 Given an array of intervals $intervals$ where $intervals[i] = [start_i, end_i]$, return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

 

Example 1:

Input: intervals = $[[1,2],[2,3],[3,4],[1,3]]$
Output: $1$
Explanation: $[1,3]$ can be removed and the rest of the intervals are non-overlapping. 
<details>
<summary>Approach</summary>
<ul>
In diary :)
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(), [](auto &x, auto &y){ return (x[1] != y[1] ? x[1] < y[1] : x[0] > y[0]);});
        vector<vector<int>> ans;
        ans.push_back(intervals[0]);
        for (int i = 1; i < (int)intervals.size(); ++i) {
            vector<int> prev = ans.back();
            if (intervals[i][0] >= prev[1]) {
                ans.push_back(intervals[i]);
            }
        }
        return (int) intervals.size() - ans.size();
    }
};
```

</ul>
</details>

</ul>
</details>
    
    
<details>
<summary>Meeting Rooms</summary>
<ul>
Given an array of meeting time intervals consisting of $start$ and $end$ times $[[s_1,e_1],[s_2,e_2],...]$ $(s_i < e_i)$, determine if a person could attend all meetings.
Input: intervals = $[(0,30),(5,10),(15,20)]$
Output: $false$
Explanation: 
$(0,30), (5,10)$ and $(0,30),(15,20)$ will conflict                                                                                                                                                                                            
                                                                                                                         
<details>
<summary>Approach</summary>
<ul>
Similar to non - overlapping intervals :)
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
bool canAttendMeetings(vector<Interval> &intervals) {
        vector<vector<int>> v;
        for (auto it: intervals) {
            v.push_back({{it.start}, {it.end}});
        }
        if (v.empty()) return true;
        sort(v.begin(), v.end(), [](auto &x, auto &y) {return x[1] != y[1] ? x[0] < y[0] : x[1] < y[1];});
        vector<vector<int>> ans;
        ans.push_back(v[0]);
        for (int i = 1; i < (int) v.size(); ++i) {
            vector<int> prev = ans.back();
            if (v[i][0] < prev[1]) return false;
            ans.push_back(v[i]);
        }
        return true;
    }
```

</ul>
</details>

</ul>
</details>

    
<details>
<summary>Meeting Rooms II</summary>
<ul>
 Given an array of meeting time intervals consisting of start and end times $[[s_1,e_1],[s_2,e_2],...]$ $(s_i < e_i)$, find the minimum number of conference rooms required.)
Input: intervals = $[(0,30),(5,10),(15,20)]$
Output: $2$
Explanation:
We need two meeting rooms
room1: $(0,30)$
room2: $(5,10),(15,20)$                                                                                                            
<details>
<summary>Approach</summary>
<ul>
Prerequisite: minheap:)
    Here, we first sort the intervals according to start time so that early meetings can be accessed earlier. Then, we maintain a min-heap and store the last meetings end time. If in $i$th meeting's start time is less than heap's end time that mean previous meeting hadn't been ended yet. So, we need a new meeting room, and thus we push new meeting's end time. Otherwise we can pop our previous meeting's end time from our $priorityqueue$ :)
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
// https://www.lintcode.com/problem/919/
/**
 * Definition of Interval:
 * classs Interval {
 *     int start, end;
 *     Interval(int start, int end) {
 *         this->start = start;
 *         this->end = end;
 *     }
 * }
 */

class Solution {
public:
    /**
     * @param intervals: an array of meeting time intervals
     * @return: the minimum number of conference rooms required
     */
    int minMeetingRooms(vector<Interval> &intervals) {
        vector<vector<int>> v;
        for (auto it: intervals) {
            v.push_back({{it.start}, {it.end}});
        }
        sort(v.begin(), v.end());
        priority_queue<int> pq;
        for (auto it: v) {
            cout << it[0] << " " << it[1] << '\n';
        }
        for (auto interval: v) {
            if (pq.empty() or -pq.top() > interval[0]) {
                pq.push(-interval[1]);
            } else {
                pq.pop();
                pq.push(-interval[1]);
            }
        }
        return (int) pq.size();
    }
};
```

</ul>
</details>

</ul>
</details>

    
    
<details>
<summary>Reverse Linked List</summary>
<ul>
    Given the head of a singly linked list, reverse the list, and return the reversed list.

![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg) 

Example 1:


Input: head = $[1,2,3,4,5]$
Output: $[5,4,3,2,1]$
<details>
<summary>Approach</summary>
<ul>
Here, main objective is to change the direction of the pointer.
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *prev, *nxt, *current;
        current = head;
        prev = nullptr;
        while (current != nullptr) {
            nxt = current -> next;
            current -> next = prev;
            prev = current;
            current = nxt;
        }
        return prev;
    }
};
```

</ul>
</details>

</ul>
</details>    
    
    
<details>
<summary>Linked List Cycle</summary>
<ul>
Given $head$, the head of a linked list, determine if the linked list has a $cycle$ in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the $next$ pointer. Internally, $pos$ is used to denote the index of the node that tail's next pointer is connected to. Note that $pos$ is not passed as a parameter.

Return $true$ if there is a cycle in the linked list. Otherwise, return $false$.  

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)    
Input: head = $[3,2,0,-4]$, pos = $1$
Output: $true$
    
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).   

<details>
<summary>Approach</summary>
<ul>
If a node appears twice in the linked list then there is surely a cycle. We can use a unordered map to check if a node already exists or not?
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        unordered_map<ListNode* , bool> mp;
        ListNode* temp = head;
        while (temp != NULL) {
            if (mp.count(temp)) return true;
            mp.insert({temp, true});
            temp = temp -> next;
        }
        return false;
    }
};
```

</ul>
</details>

</ul>
</details>
    
    
<details>
<summary>Merge Two Sorted Lists</summary>
<ul>
You are given the heads of two sorted linked lists $list1$ and $list2$.

Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list. 
    
![](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)    
Input: list1 = $[1,2,4]$, list2 = $[1,3,4]$
Output: $[1,1,2,3,4,4]$   
<details>
<summary>Approach</summary>
<ul>

> Method 1: We create a new $dummy$ node which will point to our final $ans$ linked list. Then we iterate both the linked list. If $first -> val$ is less than $second -> val$ then we create a new node whose value is $first -> val$. Our $ans$ linked list now points to our new node. Similarly we do it for $second -> val$. If one list has already ended then we can take other linked list as it is.
    
> Method 2: We will use recursive method. We don't need to create new dummy node. Here, base case is when one list is empty then we can return other list.
Or, if $list1 -> val$ is less or equal $list2 -> val$ then we call the $list1$ linked list or vice - versa.

</ul>
</details>

<details>
<summary>Method 1</summary>
<ul>
    
```c++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode *first = list1, *second = list2, *dummy = new ListNode(-101);
        ListNode *ans = dummy;
        if (first == NULL) return second;
        if (second == NULL) return first;
        while (first != NULL and second != NULL) {
            ListNode* temp;
            if (first -> val <= second -> val) {
                temp = new ListNode(first -> val);
                first = first -> next;
                ans -> next = temp;
                ans = temp;
            } else {
                temp = new ListNode(second -> val);
                second = second -> next;
                ans -> next = temp;
                ans = temp;
            }   
        }
        while (first != NULL) {
            ListNode* temp = new ListNode(first -> val);
            ans -> next = temp;
            ans = temp;
            first = first -> next;
        } 
        while (second != NULL) {
            ListNode* temp = new ListNode(second -> val);
            ans -> next = temp;
            ans = temp;
            second = second -> next;
        }
        return dummy -> next;
    }
};
```

</ul>
</details>

 <details>
<summary>Method 2</summary>
<ul>
    
```c++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if (list1 == NULL) return list2;
        else if (list2 == NULL) return list1;
        if (list1 -> val <= list2 -> val) {
            list1 -> next = mergeTwoLists(list1 -> next, list2);
            return list1;
        } else {
            list2 -> next = mergeTwoLists(list1, list2 -> next);
            return list2;
        }
    }
};    
```

</ul>
</details>
   
</ul>
</details>

    
 <details>
<summary>Longest Palindromic Substring</summary>
<ul>
  Given a string $s$, return the longest palindromic substring in $s$.  
Example 1:

Input: s = $"babad"$
Output: $"bab"$
Explanation: $"aba"$ is also a valid answer.
<details>
<summary>Approach</summary>
<ul>
Here, $O(n^2)$ solution would be expanding from the $middle$. There are two cases.
    
- case: $1$ When we've odd length of palindroms $(....cbabc...)$ then we iterate through the $mid$ element and see how far we can go to left and to right. 
    if $0 <= mid - x$ and $mid + x < n$ then we are inside the string and if both $mid - x$ and $mid + x$ character match each other then we can add this substring to our answer.
    
- case: $2$ When we've even length of palindroms $(....cbaabc...)$ then we iterate through the $mid$ element and see how far we can go to left and to right. 
    if $0 <= mid - x + 1$ and $mid + x < n$ then we are inside the string. Say our string is $baab$ and we are at $1st$ index. Now if $x = 1$ it will check 
    $s[1 - 1 + 1] == s[1 + 1]$. Similarly we check for bigger substrings that match the description.
    
</ul>
</details>

<details>
<summary>Code</summary>
<ul>
    
```c++
class Solution {
public:
    string longestPalindrome(string s) {
        int ma = 0;
        string ans;
        int n = s.size();
        for (int mid = 0; mid < n; ++mid) {
            for (int x = 0; 0 <= mid - x and mid + x < n; ++x) {
                if (s[mid - x] != s[mid + x]) break;
                int len = 2 * x + 1;
                if (len > ma) {
                    ma = len;
                    ans = s.substr(mid - x, len);
                }
            }
        }

        for (int mid = 0; mid < n; ++mid) {
            for (int x = 0; 0 <= mid - x + 1 and mid + x < n; ++x) {
                if (s[mid - x + 1] != s[mid + x]) break;
                int len = 2 * x;
                if (len > ma) {
                    ma = len;
                    ans = s.substr(mid - x + 1, len);
                }
            }
        }
        return ans;
    }
};
```

</ul>
</details>

</ul>
</details>   
    
    
    
