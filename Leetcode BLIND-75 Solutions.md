
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
        vector<vector<int>> ans;
        int n = nums.size();
        for (int k = 0; k < n; ++k) {
            while(k and k < n and nums[k] == nums[k - 1]) k++;
            int i = k + 1, j = n - 1;
            while (i < j) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum > 0) j--;
                else if (sum < 0) i++;
                else {
                   ans.push_back({nums[k], nums[i], nums[j]});
                   i++;
                   while (i < j and nums[i] == nums[i - 1]) i++; 
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
