dic={"A" : 1,"B": 10,"C":100,"D":1000,"E":10000,"F":100000,"G":1000000}
sum=0
a = input("Enter the first string: ")
for i in range(len(a)):
  if (a[i] in ["A","B","C","D","E","F","G"] ):
      sum=sum+dic[a[i]]
print(sum)
