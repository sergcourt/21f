       �K"	  ���Abrain.Event:2���M�      ��	/]���A"��
t
input/PlaceholderPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
layer_1/weights1/readIdentitylayer_1/weights1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
!layer_1/biases1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
�
layer_1/biases1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
layer_1/biases1/readIdentitylayer_1/biases1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@layer_2/weights2*
valueB"      
�
/layer_2/weights2/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/weights2*
valueB
 *׳]�
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 *
dtype0*
_output_shapes

:
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_2/weights2
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
layer_2/weights2
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
layer_2/weights2/readIdentitylayer_2/weights2*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
!layer_2/biases2/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
�
layer_2/biases2
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_2/biases2/readIdentitylayer_2/biases2*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*
T0*'
_output_shapes
:���������
S
layer_2/ReluRelulayer_2/add*'
_output_shapes
:���������*
T0
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_3/weights3*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_3/weights3/Initializer/random_uniform/minConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/maxConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3*
seed2 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
layer_3/weights3
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
layer_3/weights3/readIdentitylayer_3/weights3*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
layer_3/biases3
VariableV2*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
z
layer_3/biases3/readIdentitylayer_3/biases3*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*
T0*'
_output_shapes
:���������
S
layer_3/ReluRelulayer_3/add*
T0*'
_output_shapes
:���������
�
0output/weights4/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@output/weights4*
valueB"      
�
.output/weights4/Initializer/random_uniform/minConst*"
_class
loc:@output/weights4*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@output/weights4*
valueB
 *qĜ?
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
output/weights4
VariableV2*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
~
output/weights4/readIdentityoutput/weights4*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
 output/biases4/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
output/biases4
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
g

output/addAddoutput/MatMuloutput/biases4/read*
T0*'
_output_shapes
:���������
s
cost/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*'
_output_shapes
:���������*
T0
[

cost/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/cost/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
_output_shapes
: *
T0
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*
T0*'
_output_shapes
:���������
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
_output_shapes
:*
T0*
out_type0
q
'train/gradients/output/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_3/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: *
dtype0*
_output_shapes
: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam
VariableV2*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container 
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_2/weights2*
valueB*    
�
train/layer_2/weights2/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam
VariableV2*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container 
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container 
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
,train/output/weights4/Adam/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam_1
VariableV2*"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
+train/output/biases4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container 
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
-train/output/biases4/Adam_1/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
]
train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_2/weights2*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@output/biases4*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking( 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*
dtype0*
_output_shapes
: *%
valueB Blogging/current_cost
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
�
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"��բ�     ��d]	�I ��AJ܉
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.12.02v1.12.0-0-ga6d8ffae09��
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_1/weights1*
seed2 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
VariableV2*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_1/weights1/readIdentitylayer_1/weights1*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
layer_1/biases1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container 
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
layer_1/biases1/readIdentitylayer_1/biases1*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_2/weights2*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_2/weights2/Initializer/random_uniform/minConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
layer_2/weights2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_2/weights2/readIdentitylayer_2/weights2*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
!layer_2/biases2/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
�
layer_2/biases2
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*'
_output_shapes
:���������*
T0
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_3/weights3*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_3/weights3/Initializer/random_uniform/minConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_3/weights3*
valueB
 *׳]?
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
layer_3/weights3
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_3/weights3/readIdentitylayer_3/weights3*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
layer_3/biases3
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
z
layer_3/biases3/readIdentitylayer_3/biases3*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*
T0*'
_output_shapes
:���������
S
layer_3/ReluRelulayer_3/add*
T0*'
_output_shapes
:���������
�
0output/weights4/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@output/weights4*
valueB"      
�
.output/weights4/Initializer/random_uniform/minConst*"
_class
loc:@output/weights4*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/maxConst*"
_class
loc:@output/weights4*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
T0*"
_class
loc:@output/weights4*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
output/weights4
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
~
output/weights4/readIdentityoutput/weights4*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
g

output/addAddoutput/MatMuloutput/biases4/read*'
_output_shapes
:���������*
T0
s
cost/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*'
_output_shapes
:���������*
T0
[

cost/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/cost/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*'
_output_shapes
:���������*
T0
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/output/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
_output_shapes
:*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul*'
_output_shapes
:���������
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*'
_output_shapes
:���������*
T0
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
_output_shapes
:*
T0*
out_type0
r
(train/gradients/layer_3/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
_output_shapes
:*
T0*
out_type0
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
�
train/layer_1/weights1/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
�
train/layer_1/biases1/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_2/weights2*
valueB*    
�
train/layer_2/weights2/Adam
VariableV2*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
�
train/layer_2/biases2/Adam
VariableV2*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container 
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam
VariableV2*#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam_1
VariableV2*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
,train/output/weights4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
�
train/output/weights4/Adam
VariableV2*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam_1
VariableV2*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
+train/output/biases4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
-train/output/biases4/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container 
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_2/weights2
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@layer_3/biases3
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*!
_class
loc:@output/biases4
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking( 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*%
valueB Blogging/current_cost*
dtype0*
_output_shapes
: 
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign""
train_op


train/Adam"�
	variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
�
train/layer_1/weights1/Adam:0"train/layer_1/weights1/Adam/Assign"train/layer_1/weights1/Adam/read:02/train/layer_1/weights1/Adam/Initializer/zeros:0
�
train/layer_1/weights1/Adam_1:0$train/layer_1/weights1/Adam_1/Assign$train/layer_1/weights1/Adam_1/read:021train/layer_1/weights1/Adam_1/Initializer/zeros:0
�
train/layer_1/biases1/Adam:0!train/layer_1/biases1/Adam/Assign!train/layer_1/biases1/Adam/read:02.train/layer_1/biases1/Adam/Initializer/zeros:0
�
train/layer_1/biases1/Adam_1:0#train/layer_1/biases1/Adam_1/Assign#train/layer_1/biases1/Adam_1/read:020train/layer_1/biases1/Adam_1/Initializer/zeros:0
�
train/layer_2/weights2/Adam:0"train/layer_2/weights2/Adam/Assign"train/layer_2/weights2/Adam/read:02/train/layer_2/weights2/Adam/Initializer/zeros:0
�
train/layer_2/weights2/Adam_1:0$train/layer_2/weights2/Adam_1/Assign$train/layer_2/weights2/Adam_1/read:021train/layer_2/weights2/Adam_1/Initializer/zeros:0
�
train/layer_2/biases2/Adam:0!train/layer_2/biases2/Adam/Assign!train/layer_2/biases2/Adam/read:02.train/layer_2/biases2/Adam/Initializer/zeros:0
�
train/layer_2/biases2/Adam_1:0#train/layer_2/biases2/Adam_1/Assign#train/layer_2/biases2/Adam_1/read:020train/layer_2/biases2/Adam_1/Initializer/zeros:0
�
train/layer_3/weights3/Adam:0"train/layer_3/weights3/Adam/Assign"train/layer_3/weights3/Adam/read:02/train/layer_3/weights3/Adam/Initializer/zeros:0
�
train/layer_3/weights3/Adam_1:0$train/layer_3/weights3/Adam_1/Assign$train/layer_3/weights3/Adam_1/read:021train/layer_3/weights3/Adam_1/Initializer/zeros:0
�
train/layer_3/biases3/Adam:0!train/layer_3/biases3/Adam/Assign!train/layer_3/biases3/Adam/read:02.train/layer_3/biases3/Adam/Initializer/zeros:0
�
train/layer_3/biases3/Adam_1:0#train/layer_3/biases3/Adam_1/Assign#train/layer_3/biases3/Adam_1/read:020train/layer_3/biases3/Adam_1/Initializer/zeros:0
�
train/output/weights4/Adam:0!train/output/weights4/Adam/Assign!train/output/weights4/Adam/read:02.train/output/weights4/Adam/Initializer/zeros:0
�
train/output/weights4/Adam_1:0#train/output/weights4/Adam_1/Assign#train/output/weights4/Adam_1/read:020train/output/weights4/Adam_1/Initializer/zeros:0
�
train/output/biases4/Adam:0 train/output/biases4/Adam/Assign train/output/biases4/Adam/read:02-train/output/biases4/Adam/Initializer/zeros:0
�
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0"'
	summaries

logging/current_cost:0"�
trainable_variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08����(       �pJ	/��A*

logging/current_cost�<=�E*       ����	N=��A*

logging/current_cost�N-=�G�,*       ����	�n��A
*

logging/current_costDZ#=�)z*       ����	����A*

logging/current_costԬ=��*       ����	����A*

logging/current_cost�=���k*       ����	����A*

logging/current_cost��=�َ\*       ����	�%��A*

logging/current_cost0�=�a��*       ����	?S��A#*

logging/current_cost�O=O��*       ����	����A(*

logging/current_costV=�lh,*       ����	���A-*

logging/current_cost�: =�c(*       ����	����A2*

logging/current_costY[�<���*       ����	�	��A7*

logging/current_cost ��<��ً*       ����	8��A<*

logging/current_cost5��<((�*       ����	re��AA*

logging/current_cost�c�<����*       ����	����AF*

logging/current_cost�E�<���m*       ����	���AK*

logging/current_costKt�<2u%�*       ����	=���AP*

logging/current_cost���<�hB3*       ����	� ��AU*

logging/current_costk��<Ģ�Q*       ����	qM��AZ*

logging/current_cost��<z9HY*       ����	{��A_*

logging/current_cost�c�<Gg�*       ����	����Ad*

logging/current_cost�8�<_���*       ����	 ���Ai*

logging/current_cost���<�Ƃ*       ����	���An*

logging/current_cost���<�3˖*       ����	K3��As*

logging/current_cost�&�<3��*       ����	�a��Ax*

logging/current_cost<��<S'�Q*       ����	���A}*

logging/current_cost9��<�=}P+       ��K	���A�*

logging/current_cost���<�c� +       ��K	����A�*

logging/current_cost禧<I�%+       ��K	���A�*

logging/current_cost�i�<���+       ��K	aF��A�*

logging/current_coste�<�§/+       ��K	}r��A�*

logging/current_cost�<EB�'+       ��K	����A�*

logging/current_cost�#�<f��+       ��K	N���A�*

logging/current_cost`��<t#��+       ��K	Q	��A�*

logging/current_cost��<��+       ��K	9-	��A�*

logging/current_costVr�<�tR�+       ��K	\	��A�*

logging/current_cost�#w< �>r+       ��K	�	��A�*

logging/current_costHk<��Xm+       ��K	��	��A�*

logging/current_cost.�`<D#�6+       ��K	��	��A�*

logging/current_costwQW<�|C+       ��K	@
��A�*

logging/current_cost�+N<��Ȃ+       ��K	bA
��A�*

logging/current_cost@�E<����+       ��K	7q
��A�*

logging/current_cost��=<O��/+       ��K	՟
��A�*

logging/current_costiP6<`5�+       ��K	�
��A�*

logging/current_cost�Y/<!c�+       ��K	��
��A�*

logging/current_cost��(<U:Wl+       ��K	�(��A�*

logging/current_cost[�"<���+       ��K	�V��A�*

logging/current_cost.<�2z�+       ��K	|���A�*

logging/current_cost�V<�Ո�+       ��K	H���A�*

logging/current_costd0<ln4+       ��K	���A�*

logging/current_cost9�<���+       ��K	���A�*

logging/current_cost Q<�Q.+       ��K	�?��A�*

logging/current_cost�o
<6y�8+       ��K	l��A�*

logging/current_cost�<�O{+       ��K	���A�*

logging/current_cost0<��,?+       ��K	���A�*

logging/current_cost�~<3�+       ��K	*���A�*

logging/current_cost�3<��B+       ��K	'��A�*

logging/current_costT<���+       ��K	�T��A�*

logging/current_cost�%<��0g+       ��K	����A�*

logging/current_cost�V <����+       ��K	в��A�*

logging/current_cost~Y�;a8դ+       ��K	"���A�*

logging/current_costwJ�;�`�U+       ��K	���A�*

logging/current_costno�;%\��+       ��K	�9��A�*

logging/current_cost���;U��+       ��K	�g��A�*

logging/current_costR�;1��+       ��K	.���A�*

logging/current_cost�}�;Y#k&+       ��K	����A�*

logging/current_cost.
�;Bmq�+       ��K	����A�*

logging/current_cost��;����+       ��K	k��A�*

logging/current_cost�`�;���`+       ��K	�L��A�*

logging/current_cost%�;��ys+       ��K	�y��A�*

logging/current_cost	��;ޯ
�+       ��K	���A�*

logging/current_cost���;?;�Y+       ��K	S���A�*

logging/current_cost��;CΫ�+       ��K	�#��A�*

logging/current_cost��;wI��+       ��K	j��A�*

logging/current_cost��;XXIj+       ��K	ϭ��A�*

logging/current_cost���;��$+       ��K	����A�*

logging/current_cost���;X��+       ��K	�#��A�*

logging/current_cost���;m�+       ��K	�Y��A�*

logging/current_cost�m�;#��+       ��K	{���A�*

logging/current_cost��;���.+       ��K	X���A�*

logging/current_cost���;�ʣ�+       ��K	}��A�*

logging/current_cost\m�;��IN+       ��K	�^��A�*

logging/current_cost�;�;I8��+       ��K	ї��A�*

logging/current_cost��;x_l�+       ��K	����A�*

logging/current_costW��;L��+       ��K	���A�*

logging/current_cost���;t�":+       ��K	�8��A�*

logging/current_cost���;��5+       ��K	�s��A�*

logging/current_cost��;�F�V+       ��K	.���A�*

logging/current_costw��;��1�+       ��K	����A�*

logging/current_cost ��;f^:�+       ��K	���A�*

logging/current_cost���;I.�<+       ��K	:H��A�*

logging/current_cost���;�Te�+       ��K	�|��A�*

logging/current_cost���;7��K+       ��K	ɬ��A�*

logging/current_cost���;��	�+       ��K	����A�*

logging/current_cost��;9�Ţ+       ��K	���A�*

logging/current_cost5��;tԌ"+       ��K	�=��A�*

logging/current_cost.��;�F�[+       ��K	�l��A�*

logging/current_cost���;�[�4+       ��K	Ȟ��A�*

logging/current_cost\��;��&`+       ��K	=���A�*

logging/current_cost���;RK��+       ��K	D��A�*

logging/current_cost��;�{c�+       ��K	�2��A�*

logging/current_cost���;o���+       ��K	�g��A�*

logging/current_costW��;�Q��+       ��K	x���A�*

logging/current_cost���;�o�+       ��K	D���A�*

logging/current_costۡ�;W[,+       ��K	�!��A�*

logging/current_cost��;�E@�+       ��K	 l��A�*

logging/current_cost���; �+       ��K	F���A�*

logging/current_cost\��;R��+       ��K	����A�*

logging/current_cost���;��t+       ��K	R��A�*

logging/current_cost���;;;�&+       ��K	 K��A�*

logging/current_cost$��;����+       ��K	@}��A�*

logging/current_cost ��;�N�+       ��K	g���A�*

logging/current_costi��;]~��+       ��K	F���A�*

logging/current_costd��;0q ^+       ��K	t%��A�*

logging/current_cost���;3+       ��K	~g��A�*

logging/current_cost �;�%�+       ��K	2���A�*

logging/current_cost���;O+       ��K	����A�*

logging/current_cost���;=��&+       ��K	���A�*

logging/current_costu��;0+       ��K	tK��A�*

logging/current_cost ��;���2+       ��K	����A�*

logging/current_cost%��;��~	+       ��K	���A�*

logging/current_cost���;��+       ��K	����A�*

logging/current_costi��;��+       ��K	���A�*

logging/current_costy��;J�`+       ��K	_C��A�*

logging/current_costT��;����+       ��K	�x��A�*

logging/current_cost���;�r	�+       ��K	j���A�*

logging/current_costٯ�;�)�+       ��K	���A�*

logging/current_cost���;<iݴ+       ��K	���A�*

logging/current_cost{��;h�\�+       ��K	�5��A�*

logging/current_cost��;S���+       ��K	�b��A�*

logging/current_costn~�;����+       ��K	u���A�*

logging/current_cost�m�;�g+       ��K	����A�*

logging/current_cost[~�;�m^+       ��K	|���A�*

logging/current_cost�Z�;���B+       ��K	�/��A�*

logging/current_cost$N�;�c�`+       ��K	�_��A�*

logging/current_cost�Q�;MG�+       ��K	���A�*

logging/current_coste>�;}л�+       ��K	���A�*

logging/current_cost5:�;bd[c+       ��K	
��A�*

logging/current_cost�7�;^�i +       ��K	�=��A�*

logging/current_cost��;�Z>�+       ��K	x��A�*

logging/current_cost.%�;_�h@+       ��K	`���A�*

logging/current_cost��;��"�+       ��K	\���A�*

logging/current_cost	��;MD�M+       ��K	�?��A�*

logging/current_cost��;�[BK+       ��K	���A�*

logging/current_cost��;U(:G+       ��K	a ��A�*

logging/current_cost���;�:�[+       ��K	_[ ��A�*

logging/current_cost�;mW<�+       ��K	Ŝ ��A�*

logging/current_cost���;-~$�+       ��K	� ��A�*

logging/current_cost<��;�7~�+       ��K	�	!��A�*

logging/current_cost���;7~+       ��K	T>!��A�*

logging/current_cost���;>�J�+       ��K	�|!��A�*

logging/current_cost���;��?�+       ��K	��!��A�*

logging/current_costK��;J-F:+       ��K	�!��A�*

logging/current_cost���;�6�+       ��K	l("��A�*

logging/current_cost���;u�E�+       ��K	c"��A�*

logging/current_cost���;�c��+       ��K	�"��A�*

logging/current_costő�;�v�^+       ��K	��"��A�*

logging/current_cost�;�^�m+       ��K	&#��A�*

logging/current_cost�}�; �O�+       ��K	�=#��A�*

logging/current_cost+k�;����+       ��K	�r#��A�*

logging/current_costiu�;��#�+       ��K	��#��A�*

logging/current_cost�N�;�4��+       ��K	h$��A�*

logging/current_cost2a�;�w+N+       ��K	�W$��A�*

logging/current_cost)2�;�LA�+       ��K	��$��A�*

logging/current_costiL�;�G��+       ��K	$�$��A�*

logging/current_cost�%�;�N�&+       ��K	0%��A�*

logging/current_cost�%�;���+       ��K	�T%��A�*

logging/current_cost�2�;�~+       ��K	��%��A�*

logging/current_costE��;=�+       ��K	��%��A�*

logging/current_cost�;����+       ��K	M�%��A�*

logging/current_cost7��;�~KC+       ��K	8&��A�*

logging/current_costP��;���+       ��K	�g&��A�*

logging/current_cost���;��&S+       ��K	��&��A�*

logging/current_cost���;d|��+       ��K	��&��A�*

logging/current_cost���;�;�>+       ��K	&'��A�*

logging/current_cost`��;��5�+       ��K	�B'��A�*

logging/current_cost��;WEQ�+       ��K	r'��A�*

logging/current_costE��;�:�+       ��K	Q�'��A�*

logging/current_costp��;:(��+       ��K	��'��A�*

logging/current_cost���;|y��+       ��K	](��A�*

logging/current_cost	��;�Y+       ��K	�G(��A�*

logging/current_cost�{�;��j+       ��K	�z(��A�*

logging/current_cost���;	�+       ��K	��(��A�*

logging/current_cost�}�;P�H+       ��K	�(��A�*

logging/current_costUw�;��eG+       ��K	)��A�*

logging/current_costw}�;�.��+       ��K	K)��A�*

logging/current_cost�L�; ���+       ��K	��)��A�*

logging/current_cost<`�;k��+       ��K	;�)��A�*

logging/current_cost�U�;/:g+       ��K	:�)��A�*

logging/current_cost�K�;"M�v+       ��K	�!*��A�*

logging/current_cost�?�;Π��+       ��K	�S*��A�*

logging/current_cost�X�;�I>+       ��K	I�*��A�*

logging/current_cost7�;0��+       ��K	��*��A�*

logging/current_cost�K�;Tݪ+       ��K	M�*��A�*

logging/current_cost�6�;���+       ��K	�.+��A�*

logging/current_cost�$�;�J+       ��K	3c+��A�*

logging/current_cost�B�;��E�+       ��K	ܛ+��A�*

logging/current_costd!�;?S�p+       ��K	��+��A�*

logging/current_cost�7�;����+       ��K	,��A�*

logging/current_cost�:�;�eIA+       ��K	{D,��A�*

logging/current_cost�7�;���+       ��K	z,��A�*

logging/current_cost� �;a�Q�+       ��K	��,��A�*

logging/current_cost�.�;	��+       ��K	�-��A�*

logging/current_costn4�;� E�+       ��K	�2-��A�*

logging/current_cost��;(��+       ��K	Hj-��A�*

logging/current_cost�'�;u~��+       ��K	8�-��A�*

logging/current_cost�#�;��+       ��K	u�-��A�*

logging/current_cost�;�;�߭�+       ��K	/%.��A�*

logging/current_cost� �;�c�+       ��K	�Y.��A�*

logging/current_cost�8�;ڮ�s+       ��K	��.��A�*

logging/current_cost�5�;���+       ��K	��.��A�*

logging/current_cost�1�;̜�n+       ��K	D/��A�*

logging/current_cost$%�;����+       ��K	I=/��A�*

logging/current_costt%�;:u+       ��K	v/��A�*

logging/current_cost9C�;S��P+       ��K	H�/��A�*

logging/current_cost&�;�2��+       ��K	��/��A�*

logging/current_cost�4�;W�+       ��K	�'0��A�*

logging/current_cost�)�;ƤD"+       ��K	�f0��A�*

logging/current_cost�'�;����+       ��K	ؚ0��A�*

logging/current_cost��;*�ɤ+       ��K	�0��A�*

logging/current_cost��;3ݱ+       ��K	1��A�*

logging/current_cost �;N���+       ��K	Y1��A�*

logging/current_cost���;��+       ��K	��1��A�*

logging/current_cost���;�ו�+       ��K	&�1��A�*

logging/current_cost���;1:Q+       ��K	,2��A�*

logging/current_cost���;		��+       ��K	X2��A�*

logging/current_cost���;>8��+       ��K	Ô2��A�*

logging/current_cost���;茄+       ��K	��2��A�*

logging/current_cost+��;q�%+       ��K	)#3��A�*

logging/current_cost��;FĬ+       ��K	Qj3��A�*

logging/current_cost��;R�+       ��K	��3��A�*

logging/current_costB��;�F��+       ��K	r�3��A�	*

logging/current_cost[��;�}�+       ��K	B4��A�	*

logging/current_cost��;}G�+       ��K	�J4��A�	*

logging/current_cost�j�;v�;+       ��K	�4��A�	*

logging/current_cost��;�;K)+       ��K	��4��A�	*

logging/current_cost��;uʞ�+       ��K	X�4��A�	*

logging/current_cost�n�;��	S+       ��K	�;5��A�	*

logging/current_coste��;��+       ��K	:�5��A�	*

logging/current_cost�O�;\b��+       ��K	-�5��A�	*

logging/current_costE�;йӆ+       ��K	��5��A�	*

logging/current_cost�K�;CR��+       ��K	�;6��A�	*

logging/current_cost�m�;�=�+       ��K	*p6��A�	*

logging/current_cost\2�;Q�+       ��K	T�6��A�	*

logging/current_cost�n�;���+       ��K	��6��A�	*

logging/current_cost`3�;����+       ��K	�7��A�	*

logging/current_cost�W�;ŝu0+       ��K	N7��A�	*

logging/current_cost�P�;���+       ��K	�7��A�	*

logging/current_cost{L�;ܑ��+       ��K	�7��A�	*

logging/current_cost+�;�,��+       ��K	�8��A�	*

logging/current_costbc�;V���+       ��K	�X8��A�	*

logging/current_costn��;j���+       ��K	��8��A�	*

logging/current_cost�B�;��1}+       ��K	C�8��A�	*

logging/current_coste��;�^l�+       ��K	7&9��A�	*

logging/current_costT=�;���+       ��K	�n9��A�	*

logging/current_costw��;�\$�+       ��K	��9��A�	*

logging/current_cost�6�;Ӵ�+       ��K	;�9��A�
*

logging/current_cost��;3e��+       ��K	�:��A�
*

logging/current_costY"�;H��|+       ��K	O:��A�
*

logging/current_costI��;^K��+       ��K	+�:��A�
*

logging/current_cost+��;`\Կ+       ��K	3�:��A�
*

logging/current_cost���;;q�r+       ��K	�;��A�
*

logging/current_costԩ�;���+       ��K	�G;��A�
*

logging/current_costK�;6�?�+       ��K	�;��A�
*

logging/current_costk��;��+       ��K	;<��A�
*

logging/current_cost7	�;�+8�+       ��K	�K<��A�
*

logging/current_cost���;�N+       ��K	��<��A�
*

logging/current_cost���;! q�+       ��K	��<��A�
*

logging/current_cost���;�;��+       ��K	�7=��A�
*

logging/current_cost��;���g+       ��K	�{=��A�
*

logging/current_cost��;��� +       ��K	m�=��A�
*

logging/current_cost���;B+       ��K	c�=��A�
*

logging/current_costn��;����+       ��K	nC>��A�
*

logging/current_cost��;�,)O+       ��K	��>��A�
*

logging/current_cost5x�;�p�+       ��K	M�>��A�
*

logging/current_cost.��;����+       ��K	M�>��A�
*

logging/current_cost��;C�i+       ��K	� ?��A�
*

logging/current_cost˄�;��Y�+       ��K		_?��A�
*

logging/current_costg��;ͅ\�+       ��K	x�?��A�
*

logging/current_cost�t�;�%��+       ��K		�?��A�
*

logging/current_cost���;����+       ��K	9@��A�
*

logging/current_cost5f�;Y��H+       ��K	�F@��A�
*

logging/current_cost���;�9��+       ��K	5z@��A�*

logging/current_costW�;� ��+       ��K	�@��A�*

logging/current_cost���;��_b+       ��K	.�@��A�*

logging/current_costB�;4v��+       ��K	�A��A�*

logging/current_cost�v�;�&f+       ��K	�WA��A�*

logging/current_cost�?�;�+       ��K	�A��A�*

logging/current_cost�w�;���+       ��K	��A��A�*

logging/current_costP5�;�2-+       ��K	U�A��A�*

logging/current_costs�;d�Vf+       ��K	�,B��A�*

logging/current_cost8�;�݇�+       ��K	�rB��A�*

logging/current_costWY�;��3%+       ��K	>�B��A�*

logging/current_cost�F�;��+       ��K	:�B��A�*

logging/current_cost��;�kW�+       ��K	�C��A�*

logging/current_coste�;�:�<+       ��K	$hC��A�*

logging/current_cost��;6�+       ��K	��C��A�*

logging/current_cost5&�;P$��+       ��K	��C��A�*

logging/current_cost6�;��x+       ��K	�AD��A�*

logging/current_cost.�;��no+       ��K	l~D��A�*

logging/current_cost �;4���+       ��K	�D��A�*

logging/current_cost&�;_�#�+       ��K	��D��A�*

logging/current_cost��;�0ٴ+       ��K	�7E��A�*

logging/current_cost�;q 4�+       ��K	�mE��A�*

logging/current_cost ,�;Z9B&+       ��K	Y�E��A�*

logging/current_cost|4�;巚D+       ��K	L�E��A�*

logging/current_cost�K�;�w)+       ��K	NF��A�*

logging/current_costU�;�� �+       ��K	�@F��A�*

logging/current_cost^C�;lo�+       ��K	|F��A�*

logging/current_cost 8�;P�%�+       ��K	m�F��A�*

logging/current_cost:�;X�F+       ��K	��F��A�*

logging/current_cost$[�;q� �+       ��K	�G��A�*

logging/current_cost�]�;Ɵ�=+       ��K	�HG��A�*

logging/current_cost`n�;��G#+       ��K	�{G��A�*

logging/current_costC�;Ed�+       ��K	8�G��A�*

logging/current_cost�:�;u�\+       ��K	��G��A�*

logging/current_costg`�;
�J�+       ��K	�H��A�*

logging/current_cost�U�;O��+       ��K	�hH��A�*

logging/current_cost��;��?�+       ��K	k�H��A�*

logging/current_cost�c�;��l�+       ��K	3�H��A�*

logging/current_cost�n�;�k�D+       ��K	�I��A�*

logging/current_cost�b�;u�s�+       ��K	�BI��A�*

logging/current_cost9��;g�X+       ��K	ڍI��A�*

logging/current_cost��;˸�+       ��K	��I��A�*

logging/current_costt��;/b��+       ��K	�J��A�*

logging/current_cost��;/�K+       ��K	JJ��A�*

logging/current_cost���;�/+       ��K	>�J��A�*

logging/current_cost��;�`�+       ��K	�K��A�*

logging/current_cost���;c��L+       ��K	�FK��A�*

logging/current_cost�C�;�ӂ�+       ��K	��K��A�*

logging/current_cost���;ڂv`+       ��K	��K��A�*

logging/current_cost�q�;=`/+       ��K	5�K��A�*

logging/current_cost� �;�0�o+       ��K	c:L��A�*

logging/current_cost�t�;4AQ+       ��K	�}L��A�*

logging/current_cost���;�dQ�+       ��K	�L��A�*

logging/current_cost`A�;�n�8+       ��K	��L��A�*

logging/current_costM�;���+       ��K	;,M��A�*

logging/current_cost.n�;�~�Q+       ��K	�`M��A�*

logging/current_cost|��;�1�+       ��K	��M��A�*

logging/current_cost���;���+       ��K	��M��A�*

logging/current_cost���;JZ�)+       ��K	��M��A�*

logging/current_cost���;i�S�+       ��K	�,N��A�*

logging/current_cost{��;C��+       ��K	ZgN��A�*

logging/current_costL��;���l+       ��K	'�N��A�*

logging/current_cost���;�O�+       ��K	9�N��A�*

logging/current_cost��;�R�+       ��K	�O��A�*

logging/current_cost���;�>P�+       ��K	�FO��A�*

logging/current_cost��;�_m+       ��K	�xO��A�*

logging/current_cost��;܆��+       ��K	ǫO��A�*

logging/current_cost�.�;�nn�+       ��K	DP��A�*

logging/current_costgv�;��+       ��K	��P��A�*

logging/current_cost��;��~�+       ��K	��P��A�*

logging/current_cost.��;P9+       ��K	�P��A�*

logging/current_cost���;@�h�+       ��K	�+Q��A�*

logging/current_costu��;w+E+       ��K	�cQ��A�*

logging/current_coste��;�
P+       ��K	��Q��A�*

logging/current_cost���;��+       ��K	2�Q��A�*

logging/current_cost.��;�*�+       ��K	�R��A�*

logging/current_costΖ�;���+       ��K	 ?R��A�*

logging/current_cost���;����+       ��K	�wR��A�*

logging/current_cost�W�;c_��+       ��K	ިR��A�*

logging/current_costl��;��+       ��K	��R��A�*

logging/current_cost@��;vY�q+       ��K	S��A�*

logging/current_cost]�;�'_]+       ��K	W]S��A�*

logging/current_cost��;���+       ��K	V�S��A�*

logging/current_cost<y�;�ف�+       ��K	��S��A�*

logging/current_cost���;�܂+       ��K	J
T��A�*

logging/current_cost�|�;3 ��+       ��K	_CT��A�*

logging/current_cost���;�m}�+       ��K	�T��A�*

logging/current_cost���;݉�N+       ��K	��T��A�*

logging/current_cost$��;��k�+       ��K	��T��A�*

logging/current_costK��;�_Fe+       ��K	6U��A�*

logging/current_cost��;yP\z+       ��K	�uU��A�*

logging/current_costS�; ���+       ��K	�U��A�*

logging/current_cost��;A�.+       ��K	�U��A�*

logging/current_cost)��;��H�+       ��K	�*V��A�*

logging/current_costή�;����+       ��K	PmV��A�*

logging/current_costį�;���+       ��K	ĦV��A�*

logging/current_cost¥�;�%��+       ��K	��V��A�*

logging/current_costW�;�Ag+       ��K	W��A�*

logging/current_cost���;U�c+       ��K	�YW��A�*

logging/current_cost��;z���+       ��K	�W��A�*

logging/current_cost�;���e+       ��K	$�W��A�*

logging/current_cost���;��+       ��K	("X��A�*

logging/current_costdH�;+���+       ��K	LeX��A�*

logging/current_costu[�;�{n�+       ��K	��X��A�*

logging/current_costk��;�6�+       ��K	��X��A�*

logging/current_costBD�;�9)�+       ��K	Y��A�*

logging/current_costz�;���+       ��K	fDY��A�*

logging/current_cost<T�;�S�+       ��K	Z�Y��A�*

logging/current_cost�h�;h��+       ��K	��Y��A�*

logging/current_cost�D�;ݪ��+       ��K	��Y��A�*

logging/current_costa�;Ob�G+       ��K	�(Z��A�*

logging/current_costK�;����+       ��K	aZ��A�*

logging/current_cost"'�;�Z:�+       ��K	˪Z��A�*

logging/current_costN��;��R+       ��K	��Z��A�*

logging/current_cost�;�QC+       ��K	z[��A�*

logging/current_costDx�;Li��+       ��K	4V[��A�*

logging/current_cost�p�;�[I.+       ��K	�[��A�*

logging/current_cost�[�;*�+       ��K	�[��A�*

logging/current_cost�M�;�F|�+       ��K	\��A�*

logging/current_cost e�;�*"+       ��K	�;\��A�*

logging/current_cost�^�;����+       ��K	q\��A�*

logging/current_cost���;�O��+       ��K	}�\��A�*

logging/current_costw��;/�+       ��K	��\��A�*

logging/current_cost�p�;cw��+       ��K	]��A�*

logging/current_costdb�;&"��+       ��K	�E]��A�*

logging/current_cost�X�;m[��+       ��K	
z]��A�*

logging/current_cost�A�;cg��+       ��K	��]��A�*

logging/current_cost�[�;�{�+       ��K	�]��A�*

logging/current_cost�[�;�dX+       ��K	�(^��A�*

logging/current_cost"u�;2�+       ��K	�g^��A�*

logging/current_cost^�;t^��+       ��K	)�^��A�*

logging/current_cost�y�;9O�=+       ��K	��^��A�*

logging/current_cost��;��-+       ��K	�_��A�*

logging/current_cost�Y�;_�w�+       ��K	�I_��A�*

logging/current_costB��;��Z+       ��K	�|_��A�*

logging/current_cost�3�;��U+       ��K	'�_��A�*

logging/current_cost���;G��J+       ��K	``��A�*

logging/current_cost@5�;��Q+       ��K	�Y`��A�*

logging/current_cost���;�)�3+       ��K	�`��A�*

logging/current_cost��;��+       ��K	��`��A�*

logging/current_costdS�;��� +       ��K	�a��A�*

logging/current_cost9��;��H+       ��K	 Qa��A�*

logging/current_cost�c�;�0�+       ��K	��a��A�*

logging/current_costޙ�;8#�b+       ��K	�a��A�*

logging/current_cost���;[�6u+       ��K	x�a��A�*

logging/current_cost�|�;c�(�+       ��K	�b��A�*

logging/current_costi��;A�4�+       ��K	�Ib��A�*

logging/current_cost���;+�a+       ��K	�zb��A�*

logging/current_cost�-�;��U+       ��K	�b��A�*

logging/current_costWc�;�=��+       ��K	E�b��A�*

logging/current_cost)�;�9N�+       ��K	@c��A�*

logging/current_costkl�;2��+       ��K	QFc��A�*

logging/current_cost^��;ϳ�P+       ��K	�~c��A�*

logging/current_costU��;dP/�+       ��K	֬c��A�*

logging/current_cost-�;T���+       ��K	��c��A�*

logging/current_cost`�;\���+       ��K	d��A�*

logging/current_costrs�;���+       ��K	�?d��A�*

logging/current_cost��;�G0�+       ��K	�rd��A�*

logging/current_costk��;����+       ��K	#�d��A�*

logging/current_cost2��;�"��+       ��K	�d��A�*

logging/current_cost���;��+       ��K	="e��A�*

logging/current_costU��;���+       ��K	'Ve��A�*

logging/current_costRx�;h=+       ��K	�e��A�*

logging/current_cost+'�;x��+       ��K	��e��A�*

logging/current_cost�Z�;����+       ��K	�&f��A�*

logging/current_cost�n�;��F+       ��K	ydf��A�*

logging/current_cost�[�;�M�B+       ��K	�f��A�*

logging/current_costw+�;�8�2+       ��K	/�f��A�*

logging/current_cost��;�:�w+       ��K	k<g��A�*

logging/current_cost9�;r��,+       ��K	�vg��A�*

logging/current_cost�k�;�zs�+       ��K	E�g��A�*

logging/current_cost.]�;��U+       ��K	 �g��A�*

logging/current_cost���;��^+       ��K	#h��A�*

logging/current_cost��;���+       ��K	FUh��A�*

logging/current_cost���;�q+       ��K	h�h��A�*

logging/current_costl��;j�>+       ��K	��h��A�*

logging/current_cost��;*�W+       ��K	��h��A�*

logging/current_costn��;��`+       ��K	�i��A�*

logging/current_cost��;���+       ��K	 _i��A�*

logging/current_costN��;b$�v+       ��K	?�i��A�*

logging/current_cost���;��
�+       ��K	,�i��A�*

logging/current_cost��;��	�+       ��K	��i��A�*

logging/current_cost��;���u+       ��K	�2j��A�*

logging/current_cost|��;�m5�+       ��K	�ij��A�*

logging/current_cost���;���+       ��K	�j��A�*

logging/current_cost@��;]�6+       ��K	�j��A�*

logging/current_cost���;D���+       ��K	�k��A�*

logging/current_costBN�;<�+       ��K	g=k��A�*

logging/current_cost��;�s�+       ��K	�ik��A�*

logging/current_cost7��;r�+       ��K	(�k��A�*

logging/current_cost���;Ǹf�+       ��K	��k��A�*

logging/current_costZ�;�|�2+       ��K	�l��A�*

logging/current_cost�_�;�!E+       ��K	�6l��A�*

logging/current_cost�'�;��}+       ��K	hgl��A�*

logging/current_cost�k�;���+       ��K	��l��A�*

logging/current_cost%��;=;��+       ��K	u�l��A�*

logging/current_cost��;D\�+       ��K		m��A�*

logging/current_cost���;��P�+       ��K	�5m��A�*

logging/current_costd��;ճ9�+       ��K	�am��A�*

logging/current_cost�D�;8���+       ��K	��m��A�*

logging/current_cost���;E�;+       ��K	<�m��A�*

logging/current_cost�3�;P� +       ��K	n��A�*

logging/current_cost)��;,���+       ��K	�@n��A�*

logging/current_cost55�;
+       ��K	)qn��A�*

logging/current_costg)�;1<��+       ��K	֨n��A�*

logging/current_costdu�;E0��+       ��K	�n��A�*

logging/current_cost$3�;�9��+       ��K	wo��A�*

logging/current_costT\�;�#��+       ��K	)To��A�*

logging/current_cost)�;]�Li+       ��K	�o��A�*

logging/current_cost��;�x�+       ��K	��o��A�*

logging/current_cost���;�T:+       ��K	-	p��A�*

logging/current_cost���;i79c+       ��K	�Hp��A�*

logging/current_cost���;G��}+       ��K	C�p��A�*

logging/current_costn��;#��+       ��K	��p��A�*

logging/current_costl��;�
[i+       ��K	�p��A�*

logging/current_cost���;a�+       ��K	�0q��A�*

logging/current_costE��;h�V�+       ��K	�jq��A�*

logging/current_cost�~�;.A��+       ��K	�q��A�*

logging/current_costy��;�'�e+       ��K	��q��A�*

logging/current_cost`�;���M+       ��K	�r��A�*

logging/current_cost���;�(s+       ��K	�Gr��A�*

logging/current_cost��;��c+       ��K	){r��A�*

logging/current_cost�g�;}O+       ��K	��r��A�*

logging/current_cost��;�ٜ+       ��K	#�r��A�*

logging/current_cost��;p���+       ��K	�>s��A�*

logging/current_cost���;iv�9+       ��K	�s��A�*

logging/current_cost5��;uHv4+       ��K	J�s��A�*

logging/current_costrp�;��0r+       ��K	��s��A�*

logging/current_coste��;���+       ��K	�7t��A�*

logging/current_cost��;3<��+       ��K	�tt��A�*

logging/current_cost���;��1�+       ��K	׻t��A�*

logging/current_cost���;o�B+       ��K	\�t��A�*

logging/current_cost��;~��+       ��K	�=u��A�*

logging/current_costd��;+�3�+       ��K	`�u��A�*

logging/current_cost��;��^+       ��K	��u��A�*

logging/current_cost�@�;��v|+       ��K	U v��A�*

logging/current_costۻ�;�7d+       ��K	1v��A�*

logging/current_costnj�;	E+       ��K	�hv��A�*

logging/current_costŃ�;эv�+       ��K	��v��A�*

logging/current_cost�?�;�Z��+       ��K	��v��A�*

logging/current_costr��;��c�+       ��K	��v��A�*

logging/current_cost���;�=(+       ��K	m=w��A�*

logging/current_cost��;�BѸ+       ��K	f|w��A�*

logging/current_cost���;3�{+       ��K	̯w��A�*

logging/current_costՐ�;n�+       ��K	�w��A�*

logging/current_cost˗�;w��-+       ��K	+x��A�*

logging/current_cost���;�`�+       ��K	�bx��A�*

logging/current_coste;�;�o+       ��K	/�x��A�*

logging/current_costTo�;�d�f+       ��K	��x��A�*

logging/current_costP�;\Y&+       ��K	��x��A�*

logging/current_costkf�;���+       ��K	�#y��A�*

logging/current_cost��;�(+       ��K	IQy��A�*

logging/current_cost���;���+       ��K	v�y��A�*

logging/current_costy��;?��+       ��K	��y��A�*

logging/current_cost���;&/�)+       ��K	��y��A�*

logging/current_cost�b�;4 ��+       ��K	  z��A�*

logging/current_cost`C�;l f�+       ��K	Rz��A�*

logging/current_cost�N�;Gv��+       ��K	p�z��A�*

logging/current_cost�|�;��;{+       ��K	�z��A�*

logging/current_costj�;��L!+       ��K	��z��A�*

logging/current_cost�"�;��U�+       ��K	 {��A�*

logging/current_cost���;�>��+       ��K	�K{��A�*

logging/current_costiX�;�&h+       ��K	W�{��A�*

logging/current_costT1�;0$+       ��K	��{��A�*

logging/current_costIB�; ��+       ��K	M|��A�*

logging/current_cost���;���7+       ��K	SK|��A�*

logging/current_cost�]�;J��+       ��K	�|��A�*

logging/current_costS�;��a�+       ��K	?�|��A�*

logging/current_cost+��;W���+       ��K	}}��A�*

logging/current_costUg�;�7�+       ��K	��}��A�*

logging/current_costd�;�0�+       ��K	g�}��A�*

logging/current_cost�i�;v�p+       ��K	�6~��A�*

logging/current_cost���;��+       ��K	lp~��A�*

logging/current_cost�^�;1��+       ��K	-�~��A�*

logging/current_cost�l�;�J��+       ��K	��~��A�*

logging/current_cost��;�S��+       ��K	D��A�*

logging/current_cost���; ��+       ��K	xH��A�*

logging/current_cost�}�;P���+       ��K	U���A�*

logging/current_cost�V�;<K�+       ��K	(���A�*

logging/current_cost2��;�H�~+       ��K	����A�*

logging/current_costҝ�;ǀ,+       ��K	�!���A�*

logging/current_cost$��;��
�+       ��K	�V���A�*

logging/current_cost��;����+       ��K	����A�*

logging/current_costd��;�o�+       ��K	P��A�*

logging/current_costi��;cN��+       ��K	����A�*

logging/current_costN��;�+       ��K	�>���A�*

logging/current_costd�;��+       ��K	@r���A�*

logging/current_cost�	�;���+       ��K	ť���A�*

logging/current_cost���;�{~�+       ��K	�؁��A�*

logging/current_cost�_�;UX��+       ��K	�+���A�*

logging/current_costGK�;upG_+       ��K	�e���A�*

logging/current_cost�I�;_s�A+       ��K	0����A�*

logging/current_costą�;"���+       ��K		ǂ��A�*

logging/current_costv�;"7�D+       ��K	+��A�*

logging/current_cost���;����+       ��K	RK���A�*

logging/current_cost`��;@;�8+       ��K	�����A�*

logging/current_costnV�;�7�+       ��K	U����A�*

logging/current_cost{V�;'+       ��K	u	���A�*

logging/current_costP��;Zp�Y+       ��K	e=���A�*

logging/current_costd��;^�%g+       ��K	�m���A�*

logging/current_cost+�;O��+       ��K	K����A�*

logging/current_cost`T�;z�c�+       ��K	ք��A�*

logging/current_costN'�;Ҧ��+       ��K	���A�*

logging/current_cost��;x�Q�+       ��K	�@���A�*

logging/current_cost��;� ��+       ��K	bv���A�*

logging/current_costۉ�;@6	�+       ��K	I����A�*

logging/current_cost�-�;4��+       ��K	�慆�A�*

logging/current_cost[��;W^��+       ��K	�#���A�*

logging/current_cost��;-��+       ��K	�V���A�*

logging/current_cost���;��[+       ��K	�����A�*

logging/current_cost��;z�F)+       ��K	�����A�*

logging/current_costې�;>Kbv+       ��K	r醆�A�*

logging/current_costL��;L�+       ��K	b���A�*

logging/current_cost`��;y�y+       ��K	hI���A�*

logging/current_cost���;i�7�+       ��K	x���A�*

logging/current_costw[�;1�eO+       ��K	�����A�*

logging/current_costGH�;\pu�+       ��K	>݇��A�*

logging/current_cost���;�0�?+       ��K	���A�*

logging/current_costg��;��J]+       ��K	�8���A�*

logging/current_cost�U�;S�+       ��K	�k���A�*

logging/current_costBa�;��	�+       ��K	X����A�*

logging/current_cost:�;W�<+       ��K	�∆�A�*

logging/current_costS�;�K+       ��K	�!���A�*

logging/current_cost���;�c�R+       ��K	�a���A�*

logging/current_cost��;�{�+       ��K	ᔉ��A�*

logging/current_cost���;�[�+       ��K	}≆�A�*

logging/current_cost \�;��+       ��K	>'���A�*

logging/current_cost>8�;���^+       ��K	b���A�*

logging/current_cost�B�;0D��+       ��K	(����A�*

logging/current_cost���;�<?�+       ��K	dЊ��A�*

logging/current_cost���;]F�+       ��K	����A�*

logging/current_cost@��;�>��+       ��K	=���A�*

logging/current_cost[��;�V �+       ��K	
t���A�*

logging/current_cost�x�;M�X+       ��K	{����A�*

logging/current_cost��;ס7�+       ��K	T틆�A�*

logging/current_cost���;�\f�+       ��K	b'���A�*

logging/current_costį�;����+       ��K	@a���A�*

logging/current_costײ�;����+       ��K	֜���A�*

logging/current_cost�{�;��p+       ��K	[Ќ��A�*

logging/current_cost���;$��+       ��K	����A�*

logging/current_costb�;�k׻+       ��K	�:���A�*

logging/current_cost���;�Ee+       ��K	�l���A�*

logging/current_cost�p�;����+       ��K	؟���A�*

logging/current_costb��;#%/+       ��K	f׍��A�*

logging/current_cost��;)�`+       ��K	���A�*

logging/current_cost�}�;��iI+       ��K	�>���A�*

logging/current_costrz�;�l��+       ��K	�q���A�*

logging/current_cost�2�;�0�P+       ��K	�����A�*

logging/current_costN��;�*��+       ��K	BЎ��A�*

logging/current_cost ��;ʅv+       ��K	�����A�*

logging/current_cost'7�;e�ͅ+       ��K	(2���A�*

logging/current_cost.]�;9"C+       ��K	`���A�*

logging/current_cost$0�;^�M+       ��K	G����A�*

logging/current_cost���;����+       ��K	ع���A�*

logging/current_cost���;"B�7+       ��K	�珆�A�*

logging/current_cost4��;�d�+       ��K	)8���A�*

logging/current_costg�;��a+       ��K	�h���A�*

logging/current_cost v�;8s�+       ��K	5����A�*

logging/current_cost'��;s��+       ��K	�Ð��A�*

logging/current_cost�_�;r��+       ��K	���A�*

logging/current_cost���;�1�+       ��K	����A�*

logging/current_cost�w�;F.D+       ��K	�M���A�*

logging/current_costW�;U�z�+       ��K	�|���A�*

logging/current_cost�a�;b,��+       ��K	s����A�*

logging/current_cost��;���>+       ��K	7ܑ��A�*

logging/current_cost�q�;;��+       ��K	����A�*

logging/current_cost�W�;��#H+       ��K	d:���A�*

logging/current_cost���;��Cu+       ��K	�h���A�*

logging/current_cost��;޷x%+       ��K	ۖ���A�*

logging/current_cost|]�;m0=�+       ��K	�Ē��A�*

logging/current_cost���;�C�+       ��K	6����A�*

logging/current_costҰ�;9��+       ��K	�)���A�*

logging/current_cost+d�;���+       ��K	RZ���A�*

logging/current_cost�S�;Q\�+       ��K	Ј���A�*

logging/current_cost\��;�	��+       ��K	L����A�*

logging/current_costTg�;y�z�+       ��K	듆�A�*

logging/current_costr�;�qL+       ��K	Q���A�*

logging/current_cost�`�;�k�+       ��K	K���A�*

logging/current_cost��;�1Gc+       ��K	g~���A�*

logging/current_cost�r�;؈ �+       ��K	�����A�*

logging/current_costkH�;���.+       ��K	�ٔ��A�*

logging/current_costn��;q�Ke+       ��K	����A�*

logging/current_cost>�;�:H(+       ��K	C6���A�*

logging/current_costRf�;���+       ��K	�g���A�*

logging/current_cost|��;[V�V+       ��K	;����A�*

logging/current_cost.��;����+       ��K	ʕ��A�*

logging/current_costL�;CS��+       ��K	)����A�*

logging/current_costE�;J2�I+       ��K	�+���A�*

logging/current_cost2��;��f�+       ��K	�\���A�*

logging/current_costΟ�;{�'�+       ��K	�����A�*

logging/current_cost���;W�j�+       ��K	����A�*

logging/current_cost���;��s+       ��K	�떆�A�*

logging/current_costK��;�7�+       ��K	����A�*

logging/current_costg��;L�Pc+       ��K	zH���A�*

logging/current_cost���;����+       ��K	h|���A�*

logging/current_costn��;�	+       ��K	5����A�*

logging/current_costN��;��O+       ��K	ٗ��A�*

logging/current_cost� �;�w'+       ��K	p	���A�*

logging/current_cost��;l��+       ��K	S7���A�*

logging/current_cost\��;bC��+       ��K	h���A�*

logging/current_cost��;�Z�N+       ��K	�����A�*

logging/current_cost�X�;�4�+       ��K	�Ř��A�*

logging/current_cost�Z�;��T�+       ��K	�����A�*

logging/current_cost��;��+       ��K	e!���A�*

logging/current_costّ�;�n��+       ��K	�M���A�*

logging/current_cost%��;ȇ��+       ��K	(}���A�*

logging/current_cost���;�M;�+       ��K	穙��A�*

logging/current_cost���;9a��+       ��K	*ؙ��A�*

logging/current_cost��;	�N�+       ��K	����A�*

logging/current_cost�<�;�>��+       ��K	�4���A�*

logging/current_costա�;���+       ��K	�`���A�*

logging/current_cost���;�Q��+       ��K	叚��A�*

logging/current_cost���;K�Q+       ��K	5����A�*

logging/current_cost��;���+       ��K	S욆�A�*

logging/current_cost)g�;���<+       ��K	1���A�*

logging/current_costL^�;@[
+       ��K	/G���A�*

logging/current_costp��;_�+       ��K	*t���A�*

logging/current_cost9��;!�8�+       ��K	����A�*

logging/current_cost�p�;^@�+       ��K	LЛ��A�*

logging/current_costt��;oz+       ��K	����A�*

logging/current_cost^��;��5+       ��K	�+���A�*

logging/current_cost5,�;�.$�+       ��K	�W���A�*

logging/current_costG.�;�jt�+       ��K	˄���A�*

logging/current_cost���;���+       ��K	����A�*

logging/current_cost��;	3o�+       ��K	2ݜ��A�*

logging/current_cost�b�;T��+       ��K	 ���A�*

logging/current_cost�q�;�uL�+       ��K	�:���A�*

logging/current_costei�;<]>+       ��K	Ng���A�*

logging/current_costde�;v6+       ��K	�����A�*

logging/current_cost�[�;$��+       ��K	���A�*

logging/current_cost��;�9�+       ��K	_��A�*

logging/current_cost<��;��z�+       ��K	, ���A�*

logging/current_cost\a�;�"�E+       ��K	M���A�*

logging/current_cost��;CVTP+       ��K	Yz���A�*

logging/current_cost��;�:e+       ��K	�����A�*

logging/current_cost+��;d���+       ��K	}ٞ��A�*

logging/current_costJ�;+cM�+       ��K	w���A�*

logging/current_cost��;�'ו+       ��K	�5���A�*

logging/current_cost`��;��+       ��K	�a���A�*

logging/current_costn$�;e�z�+       ��K	�����A�*

logging/current_cost\�;�%L�+       ��K	�ß��A�*

logging/current_cost�9�;��=c+       ��K	Z����A�*

logging/current_cost���;�ַ+       ��K	M(���A�*

logging/current_cost���;}j J+       ��K	PV���A�*

logging/current_costu�;i�@+       ��K	Ņ���A�*

logging/current_cost9p�;My�x+       ��K	g����A�*

logging/current_cost�9�;��lu+       ��K	�㠆�A�*

logging/current_cost�3�;s�fP+       ��K	����A�*

logging/current_cost�g�;�$>h+       ��K	>���A�*

logging/current_costyp�;�F�+       ��K	Dn���A�*

logging/current_costNk�;�l�N+       ��K	�����A�*

logging/current_cost���;�}N+       ��K	�ɡ��A�*

logging/current_costШ�;Nm��+       ��K	j����A�*

logging/current_cost���;Z��+       ��K	B+���A�*

logging/current_cost���;���+       ��K	U\���A�*

logging/current_cost�j�;D�+       ��K	����A�*

logging/current_cost¨�;����+       ��K	�����A�*

logging/current_cost.2�;1 �Q+       ��K	k좆�A�*

logging/current_cost^>�;0-�+       ��K	����A�*

logging/current_cost�n�;��(:+       ��K	zG���A�*

logging/current_cost��;\��+       ��K	�u���A�*

logging/current_cost���;r{�+       ��K	I����A�*

logging/current_cost5��;q��T+       ��K	�֣��A�*

logging/current_cost�G�;#
W+       ��K	0���A�*

logging/current_cost���;��Z�+       ��K	s3���A�*

logging/current_cost���;��Y�+       ��K	�g���A�*

logging/current_costK�;�:uv+       ��K	�����A�*

logging/current_costUm�;��;�+       ��K	����A�*

logging/current_cost'�;A��+       ��K	���A�*

logging/current_costg{�;����+       ��K	
!���A�*

logging/current_cost���;r���+       ��K	tP���A�*

logging/current_cost���;�L��+       ��K	k���A�*

logging/current_cost<��;�[b+       ��K	j����A�*

logging/current_cost��;��ڕ+       ��K	tݥ��A�*

logging/current_costg��;KD+       ��K	����A�*

logging/current_cost)`�;�O��+       ��K	;���A�*

logging/current_cost��;�H�\+       ��K	�i���A�*

logging/current_cost�o�;���M+       ��K	H����A�*

logging/current_cost�D�;fö\+       ��K	�Ħ��A�*

logging/current_cost�j�;���+       ��K	���A�*

logging/current_cost�S�;Ԭ��+       ��K	;#���A�*

logging/current_cost���;;Jp+       ��K	�V���A�*

logging/current_costiv�;c���+       ��K	K����A�*

logging/current_cost�l�;�o+       ��K	G����A�*

logging/current_cost���;����+       ��K	#ᧆ�A�*

logging/current_cost"��;�a��+       ��K	����A�*

logging/current_cost���;�B�+       ��K	?���A�*

logging/current_costA�;9�Sp+       ��K	ll���A�*

logging/current_cost���;�/
+       ��K	뙨��A�*

logging/current_cost��;YEY+       ��K	�ƨ��A�*

logging/current_cost�P�;�_R+       ��K	���A�*

logging/current_cost���;m��-+       ��K	�!���A�*

logging/current_cost��;v^k]+       ��K	O���A�*

logging/current_cost���;7���+       ��K	A}���A�*

logging/current_cost^��;\$jq+       ��K	;����A�*

logging/current_cost�s�;�!E@+       ��K	Xة��A�*

logging/current_cost��;z�X0+       ��K	:���A�*

logging/current_cost
�;�L��+       ��K	Q4���A�*

logging/current_cost���;$�L+       ��K	�a���A�*

logging/current_cost���;�=�Y+       ��K	&����A�*

logging/current_cost��;��31+       ��K	Q����A�*

logging/current_cost2��;5��+       ��K	d몆�A�*

logging/current_cost���;'L��+       ��K	����A�*

logging/current_costtb�;|k�+       ��K	�D���A�*

logging/current_cost���;!0+       ��K	�r���A�*

logging/current_cost���;��ٚ+       ��K	K����A�*

logging/current_costy�;b�Y�+       ��K	�Ы��A�*

logging/current_cost>��;�ǲ*+       ��K	v ���A�*

logging/current_cost��;��Z+       ��K	�,���A�*

logging/current_costdw�;����+       ��K	�[���A�*

logging/current_cost��;�kZ�+       ��K	M����A�*

logging/current_cost��;�rB�+       ��K	-����A�*

logging/current_cost���;�)0�+       ��K	d鬆�A�*

logging/current_cost���;�,=+       ��K	����A�*

logging/current_cost���;ziC+       ��K	�G���A�*

logging/current_cost~��;�!�+       ��K	�t���A�*

logging/current_costy[�;($�;+       ��K	!����A�*

logging/current_cost˫�;H�+       ��K	�ҭ��A�*

logging/current_cost��;QX+       ��K	� ���A�*

logging/current_costn#�;mĩE+       ��K	./���A�*

logging/current_cost���;���n+       ��K	_���A�*

logging/current_cost�x�;v)}+       ��K	�����A�*

logging/current_cost���;S6gB+       ��K	㻮��A�*

logging/current_cost��;�	+       ��K	U鮆�A�*

logging/current_cost�R�;���:+       ��K	y���A�*

logging/current_costU��;�3�+       ��K	�K���A�*

logging/current_costD�;��'s+       ��K	�x���A�*

logging/current_cost@j�;���+       ��K	�����A�*

logging/current_costB6�;�Ww+       ��K	�ԯ��A�*

logging/current_cost�F�;�P�6+       ��K	g���A�*

logging/current_cost$U�;��+       ��K	�5���A�*

logging/current_cost��;�:��+       ��K	�e���A�*

logging/current_cost>��;�
'�+       ��K	@����A�*

logging/current_costt��;��j+       ��K	ư��A�*

logging/current_cost$��;	અ+       ��K	�����A�*

logging/current_cost���;E)c+       ��K	(%���A�*

logging/current_cost%��;E�%�+       ��K	�U���A�*

logging/current_cost��;�ƽo+       ��K	]����A�*

logging/current_costNe�;�&+       ��K	X����A� *

logging/current_cost4>�;�,JR+       ��K	:߱��A� *

logging/current_cost`��;&F+       ��K	)���A� *

logging/current_cost ��;���+       ��K	F@���A� *

logging/current_cost�*�;���+       ��K	Xn���A� *

logging/current_cost���;ط�+       ��K	����A� *

logging/current_cost��;9�:+       ��K	�Ȳ��A� *

logging/current_cost�W�;Z���+       ��K	�����A� *

logging/current_cost�'�;��V_+       ��K	)(���A� *

logging/current_cost�O�;��+       ��K	�Y���A� *

logging/current_cost+�;d%]d+       ��K	�����A� *

logging/current_cost�Q�;h8�+       ��K	q����A� *

logging/current_cost�o�;A�a�+       ��K	�ೆ�A� *

logging/current_cost���;�B+       ��K	����A� *

logging/current_costl��;l���+       ��K	�?���A� *

logging/current_costR��;5Et�+       ��K	:o���A� *

logging/current_cost���;o���+       ��K	p����A� *

logging/current_coste�;���+       ��K	˴��A� *

logging/current_cost��;�+       ��K	����A� *

logging/current_cost�C�;k���+       ��K	b(���A� *

logging/current_cost[��;�P.�+       ��K	�Y���A� *

logging/current_cost`��;%�X�+       ��K	)����A� *

logging/current_cost.=�;�G�+       ��K	�����A� *

logging/current_cost6�;��c8+       ��K	O䵆�A� *

logging/current_costkW�;Y�b+       ��K	���A� *

logging/current_cost� �;��+       ��K	�>���A�!*

logging/current_cost�V�; ��&+       ��K	�l���A�!*

logging/current_cost��;�/hw+       ��K	+����A�!*

logging/current_cost���;�#]5+       ��K	�ɶ��A�!*

logging/current_cost���;��U\+       ��K	T����A�!*

logging/current_cost���;���+       ��K	�'���A�!*

logging/current_cost��;b�\+       ��K	�Y���A�!*

logging/current_cost�l�;.M��+       ��K	�����A�!*

logging/current_cost��;�o�+       ��K	7����A�!*

logging/current_cost�-�;�p+       ��K	�㷆�A�!*

logging/current_cost.�;S�Uq+       ��K	A���A�!*

logging/current_cost�f�;��nD+       ��K	�=���A�!*

logging/current_cost��;���o+       ��K	Nk���A�!*

logging/current_cost���;�`L+       ��K	h����A�!*

logging/current_cost\!�;�13+       ��K	�Ǹ��A�!*

logging/current_costĲ�;Z��+       ��K	=����A�!*

logging/current_cost�X�;�,��+       ��K	i ���A�!*

logging/current_cost`��;N�;+       ��K	�N���A�!*

logging/current_cost��;�H+       ��K	�}���A�!*

logging/current_costE��;�W�+       ��K	Ө���A�!*

logging/current_cost���;p��+       ��K	�׹��A�!*

logging/current_costT�;��+       ��K	?���A�!*

logging/current_cost<r�;�oV+       ��K	�B���A�!*

logging/current_costbL�;�L�+       ��K	�p���A�!*

logging/current_costR7�;t�J�+       ��K	
����A�!*

logging/current_cost.�;�=W+       ��K	ZϺ��A�!*

logging/current_cost�V�;����+       ��K	�����A�"*

logging/current_costEd�;��>�+       ��K	--���A�"*

logging/current_costR��;���+       ��K	`Y���A�"*

logging/current_cost��;]+       ��K	�����A�"*

logging/current_cost���;sT��+       ��K	����A�"*

logging/current_cost%��;���+       ��K	�K���A�"*

logging/current_cost���;�~��+       ��K	����A�"*

logging/current_cost���;�ː�+       ��K	Ỽ��A�"*

logging/current_cost���;�^�}+       ��K	�＆�A�"*

logging/current_costl�;P� z+       ��K	M ���A�"*

logging/current_cost�<�;wÓ@+       ��K	�P���A�"*

logging/current_cost�~�;��B+       ��K	�}���A�"*

logging/current_cost�*�;��6+       ��K	�����A�"*

logging/current_cost��;��E%+       ��K	�߽��A�"*

logging/current_cost�Z�;\.+       ��K	����A�"*

logging/current_cost ��;��{}+       ��K	u?���A�"*

logging/current_cost|��;
�+       ��K	�|���A�"*

logging/current_cost���;���+       ��K	�����A�"*

logging/current_cost"$�;����+       ��K	>꾆�A�"*

logging/current_cost 1�;�m׷+       ��K	���A�"*

logging/current_cost��;�?!�+       ��K	�M���A�"*

logging/current_cost���;�l��+       ��K	�{���A�"*

logging/current_cost���;�Ü�+       ��K	�����A�"*

logging/current_cost�|�;'��P+       ��K	rڿ��A�"*

logging/current_cost�1�;#$1+       ��K	����A�"*

logging/current_cost�3�;�e-N+       ��K	[;���A�#*

logging/current_cost7�;x�G+       ��K	 j���A�#*

logging/current_cost"L�;5ܴ~+       ��K	�����A�#*

logging/current_cost�q�;�i��+       ��K	�����A�#*

logging/current_cost@��;��Bg+       ��K	l����A�#*

logging/current_cost ��;Vsz�+       ��K	�)���A�#*

logging/current_cost�	�;�q��+       ��K	*\���A�#*

logging/current_costY��;�ï+       ��K	����A�#*

logging/current_cost�U�;�|H:+       ��K	�����A�#*

logging/current_cost.�;��+       ��K	�����A�#*

logging/current_costDR�;��jW+       ��K	��A�#*

logging/current_cost$�;��6+       ��K	F�A�#*

logging/current_cost9N�;�/$�+       ��K	Fy�A�#*

logging/current_costΝ�;\n��+       ��K	��A�#*

logging/current_cost���;��+       ��K	���A�#*

logging/current_cost�@�;Ǯ*�+       ��K	Æ�A�#*

logging/current_cost��;�:h�+       ��K	�6Æ�A�#*

logging/current_cost2��;��!+       ��K	�eÆ�A�#*

logging/current_cost|��;��@+       ��K	�Æ�A�#*

logging/current_cost���;���+       ��K	��Æ�A�#*

logging/current_cost+l�;���+       ��K	�Æ�A�#*

logging/current_costN�;Z��+       ��K	o0Ć�A�#*

logging/current_costg5�;�r�j+       ��K	-aĆ�A�#*

logging/current_cost��;��!+       ��K	f�Ć�A�#*

logging/current_costt~�;�Tc�+       ��K	��Ć�A�#*

logging/current_cost��;B�5+       ��K	f�Ć�A�#*

logging/current_cost���;�Ȧg+       ��K	�2ņ�A�$*

logging/current_cost�g�;>��/+       ��K	�eņ�A�$*

logging/current_cost;��;S\�+       ��K	��ņ�A�$*

logging/current_cost���;&5hL+       ��K	P�ņ�A�$*

logging/current_cost�q�;p�g3+       ��K	qƆ�A�$*

logging/current_cost^��;0��+       ��K	1Ɔ�A�$*

logging/current_cost��;VFxU+       ��K	�fƆ�A�$*

logging/current_cost+K�;��)+       ��K	{�Ɔ�A�$*

logging/current_cost�?�;�"Y^+       ��K	�Ɔ�A�$*

logging/current_cost���;����+       ��K	jǆ�A�$*

logging/current_cost̪�;q��+       ��K	�Kǆ�A�$*

logging/current_cost�t�;W/�@+       ��K	3�ǆ�A�$*

logging/current_costy��;�A7�+       ��K	|�ǆ�A�$*

logging/current_cost���;�g�+       ��K	��ǆ�A�$*

logging/current_costTb�;���+       ��K	�Ȇ�A�$*

logging/current_costiI�;�q�+       ��K	NȆ�A�$*

logging/current_cost�Y�;�x|�+       ��K	��Ȇ�A�$*

logging/current_cost\��;:&]�+       ��K	>�Ȇ�A�$*

logging/current_cost�^�;Taʕ+       ��K	��Ȇ�A�$*

logging/current_cost�8�;;c�$+       ��K	-Ɇ�A�$*

logging/current_cost���;�ٖ�+       ��K	�pɆ�A�$*

logging/current_cost���;�z+       ��K	>�Ɇ�A�$*

logging/current_cost�2�;G��)+       ��K	��Ɇ�A�$*

logging/current_cost�O�;�ĚL+       ��K	�ʆ�A�$*

logging/current_cost�-�;�v0b+       ��K	8Qʆ�A�$*

logging/current_cost�"�;Z�yk+       ��K	��ʆ�A�$*

logging/current_cost�`�;Wy�+       ��K	%�ʆ�A�%*

logging/current_cost9y�;k��+       ��K	��ʆ�A�%*

logging/current_cost	�;�ᕔ+       ��K	�1ˆ�A�%*

logging/current_costYP�;Ǵ��+       ��K	�dˆ�A�%*

logging/current_cost�3�;�`+       ��K	��ˆ�A�%*

logging/current_cost���;����+       ��K	��ˆ�A�%*

logging/current_cost���;��D]+       ��K	�̆�A�%*

logging/current_cost��;>���+       ��K	RD̆�A�%*

logging/current_costUA�;L�e�+       ��K	x̆�A�%*

logging/current_costk��;[�8@+       ��K	|�̆�A�%*

logging/current_cost$(�;�H*q+       ��K	�̆�A�%*

logging/current_cost�l�;W_v�+       ��K	Q'͆�A�%*

logging/current_cost���;W3q�+       ��K	]͆�A�%*

logging/current_cost���;�%�#+       ��K	֍͆�A�%*

logging/current_cost��;T�/�+       ��K	�͆�A�%*

logging/current_cost9�;	���+       ��K	�͆�A�%*

logging/current_cost���;0�m9+       ��K	� Ά�A�%*

logging/current_cost��;�Z�E+       ��K	wRΆ�A�%*

logging/current_cost���;�.4�+       ��K	 �Ά�A�%*

logging/current_cost\��;��v+       ��K	%�Ά�A�%*

logging/current_cost*�;m	�>+       ��K	$φ�A�%*

logging/current_cost���;����+       ��K	�6φ�A�%*

logging/current_cost0�;�u��+       ��K	�}φ�A�%*

logging/current_cost�*�;��%+       ��K	s�φ�A�%*

logging/current_cost5��;O��q+       ��K	��φ�A�%*

logging/current_costdi�;x%+       ��K	*І�A�&*

logging/current_cost[�;?�;�+       ��K	�hІ�A�&*

logging/current_cost���;�ߡ+       ��K	��І�A�&*

logging/current_cost��;�ci+       ��K	�І�A�&*

logging/current_cost�<�;@SR�+       ��K	#?ц�A�&*

logging/current_cost�E�;\m�9+       ��K	]yц�A�&*

logging/current_cost�"�;j~�+       ��K	J�ц�A�&*

logging/current_cost��;�S��+       ��K	h�ц�A�&*

logging/current_costG�;\�cZ+       ��K	|$҆�A�&*

logging/current_costD��;��@+       ��K	�V҆�A�&*

logging/current_cost��;����+       ��K	��҆�A�&*

logging/current_costQ�;�++       ��K	��҆�A�&*

logging/current_cost�*�;+�w|+       ��K	X�҆�A�&*

logging/current_cost@��;����+       ��K	�-ӆ�A�&*

logging/current_cost�+�;��s2+       ��K	�^ӆ�A�&*

logging/current_cost^;�;â/r+       ��K	Ēӆ�A�&*

logging/current_cost ��;�?�+       ��K	�ӆ�A�&*

logging/current_costuR�;s�+       ��K	�Ԇ�A�&*

logging/current_cost�Y�;���7+       ��K	�BԆ�A�&*

logging/current_cost��;?6�+       ��K	�}Ԇ�A�&*

logging/current_costܲ�;	��+       ��K	B�Ԇ�A�&*

logging/current_cost0:�;6��+       ��K	��Ԇ�A�&*

logging/current_cost�V�;��;+       ��K	_Ն�A�&*

logging/current_costR��;}��)+       ��K	�=Ն�A�&*

logging/current_cost�:�;ȏu+       ��K	�kՆ�A�&*

logging/current_cost,B�;<�Q+       ��K	g�Ն�A�&*

logging/current_costl��;��+       ��K	��Ն�A�'*

logging/current_cost���;���O+       ��K	1�Ն�A�'*

logging/current_cost�C�;��y�+       ��K	'ֆ�A�'*

logging/current_cost.N�;s]�+       ��K	Vֆ�A�'*

logging/current_costk��;��(+       ��K	`�ֆ�A�'*

logging/current_cost�N�;T	X+       ��K	��ֆ�A�'*

logging/current_cost���;X?��+       ��K	�׆�A�'*

logging/current_costNj�;o��+       ��K	F׆�A�'*

logging/current_costTd�;�z +       ��K	v׆�A�'*

logging/current_costN�;�1�.+       ��K	v�׆�A�'*

logging/current_cost�^�;����+       ��K	��׆�A�'*

logging/current_cost I�;|�4�+       ��K	.؆�A�'*

logging/current_costD��;
5�+       ��K	7؆�A�'*

logging/current_cost�6�;��7]+       ��K	f؆�A�'*

logging/current_cost0�;Ȣ�9+       ��K	��؆�A�'*

logging/current_cost��;���}+       ��K	�؆�A�'*

logging/current_cost$t�;I�+       ��K	��؆�A�'*

logging/current_costy�;*l�u+       ��K	$ن�A�'*

logging/current_cost��;)��+       ��K	#Sن�A�'*

logging/current_cost��;�53n+       ��K	�ن�A�'*

logging/current_costWb�;媄�+       ��K	9�ن�A�'*

logging/current_costU8�;jr�<+       ��K	��ن�A�'*

logging/current_cost �;��s�+       ��K	tچ�A�'*

logging/current_cost?�;9��-+       ��K	�@چ�A�'*

logging/current_cost��;�s�0+       ��K	@nچ�A�'*

logging/current_cost�N�;�
��+       ��K	r�چ�A�(*

logging/current_cost'�;��&+       ��K	]�چ�A�(*

logging/current_cost`u�;�p9+       ��K	6�چ�A�(*

logging/current_cost���;��+       ��K	�.ۆ�A�(*

logging/current_cost���;�çS+       ��K	w[ۆ�A�(*

logging/current_cost0 �;���+       ��K	<�ۆ�A�(*

logging/current_cost^>�;D�+       ��K	µۆ�A�(*

logging/current_cost9;�;mMI�+       ��K	n�ۆ�A�(*

logging/current_cost^��;a��9+       ��K	܆�A�(*

logging/current_cost$��;���+       ��K	?܆�A�(*

logging/current_cost�j�;����+       ��K	l܆�A�(*

logging/current_cost	C�;�"+       ��K	�܆�A�(*

logging/current_cost<��;.t +       ��K	��܆�A�(*

logging/current_cost�N�;Â�B+       ��K	W݆�A�(*

logging/current_cost\��;�M�+       ��K	�4݆�A�(*

logging/current_cost>|�;*qa+       ��K	�b݆�A�(*

logging/current_cost٭�;��z2+       ��K	ӏ݆�A�(*

logging/current_cost���;�D�+       ��K	��݆�A�(*

logging/current_costۣ�;bE��+       ��K	��݆�A�(*

logging/current_cost'[�;be�+       ��K	zކ�A�(*

logging/current_costD�;�مP+       ��K	ZMކ�A�(*

logging/current_cost�;%��+       ��K	�yކ�A�(*

logging/current_cost<��;�ܣb+       ��K	�ކ�A�(*

logging/current_cost7,�;�mW+       ��K	{�ކ�A�(*

logging/current_costU��;� ��+       ��K	�߆�A�(*

logging/current_costBL�;H��+       ��K	�1߆�A�(*

logging/current_costn��;#��+       ��K	>^߆�A�)*

logging/current_cost<T�;��'+       ��K	1�߆�A�)*

logging/current_cost�
�;�>_ +       ��K	E�߆�A�)*

logging/current_cost�+�;0�)H+       ��K	��߆�A�)*

logging/current_cost�;U:c�+       ��K	 ���A�)*

logging/current_costA�;P���+       ��K	dM���A�)*

logging/current_costj�;�+       ��K	�}���A�)*

logging/current_cost@��;�M�+       ��K	$����A�)*

logging/current_costJ�;��l+       ��K	�����A�)*

logging/current_costd�;-u�+       ��K	��A�)*

logging/current_cost�;܂�f+       ��K	�D��A�)*

logging/current_cost�L�;�;u�+       ��K	�u��A�)*

logging/current_cost�_�;�1e�+       ��K	����A�)*

logging/current_costk��;w")+       ��K	����A�)*

logging/current_cost{g�;7td�+       ��K	����A�)*

logging/current_cost���;my�+       ��K	�+��A�)*

logging/current_cost���;��+       ��K	�]��A�)*

logging/current_cost5��;L��+       ��K	���A�)*

logging/current_costJ�;�O�+       ��K	}���A�)*

logging/current_cost���;CU�V+       ��K	���A�)*

logging/current_costD��;���>+       ��K	���A�)*

logging/current_cost�*�;F�P�+       ��K	�>��A�)*

logging/current_cost�2�;�c��+       ��K	�r��A�)*

logging/current_costb��;���+       ��K	����A�)*

logging/current_cost��;K�2y+       ��K	@���A�)*

logging/current_costɟ�;#�P�+       ��K	����A�)*

logging/current_cost���;B��+       ��K	V-��A�**

logging/current_cost{��;~ҟ�+       ��K	�^��A�**

logging/current_cost9��;dӽ+       ��K	����A�**

logging/current_cost��;5!�(+       ��K	����A�**

logging/current_cost"w�;���+       ��K	i���A�**

logging/current_costN`�;���+       ��K	U��A�**

logging/current_cost7��;V��z+       ��K	1H��A�**

logging/current_cost��;]�+       ��K	@u��A�**

logging/current_cost4��;8��+       ��K	ܨ��A�**

logging/current_cost��;��+       ��K	����A�**

logging/current_cost|��;vC+       ��K	+��A�**

logging/current_cost��;�)�_+       ��K	3��A�**

logging/current_cost���;����+       ��K	pb��A�**

logging/current_cost��;76�+       ��K	���A�**

logging/current_cost ��;	��+       ��K	����A�**

logging/current_costK��;&��+       ��K	"���A�**

logging/current_costL��;����+       ��K	���A�**

logging/current_cost���;
��\+       ��K	�P��A�**

logging/current_cost���;���6+       ��K	�~��A�**

logging/current_cost�w�;��Ћ+       ��K	a���A�**

logging/current_cost	��;L��+       ��K	-���A�**

logging/current_costn��;>�q�+       ��K	B	��A�**

logging/current_costr��;%^+       ��K	'8��A�**

logging/current_cost@��;G/j�+       ��K	�k��A�**

logging/current_cost��;��Ua+       ��K	,���A�**

logging/current_costG�;ٓ��+       ��K	����A�+*

logging/current_cost���;eP~�+       ��K	����A�+*

logging/current_cost� �;�I+       ��K	#��A�+*

logging/current_costk�;U�tf+       ��K	SP��A�+*

logging/current_costk@�;tWz+       ��K	g���A�+*

logging/current_costb��;˶L+       ��K	����A�+*

logging/current_costy�;��T+       ��K	}���A�+*

logging/current_cost���;[��T+       ��K	���A�+*

logging/current_cost;��;w�+       ��K	�4��A�+*

logging/current_cost���;N�l�+       ��K	�`��A�+*

logging/current_costԋ�;)_	�+       ��K	n���A�+*

logging/current_cost���;��RI+       ��K	P���A�+*

logging/current_costܐ�;�nб+       ��K	 ���A�+*

logging/current_cost�/�;=��v+       ��K	���A�+*

logging/current_cost���;�w�+       ��K	F��A�+*

logging/current_cost��;W^�a+       ��K	$t��A�+*

logging/current_costG��;u=��+       ��K	(���A�+*

logging/current_cost���;G5�+       ��K	����A�+*

logging/current_cost���;��e+       ��K	&���A�+*

logging/current_cost��;�$�+       ��K	1,��A�+*

logging/current_cost>��;�D+       ��K	�X��A�+*

logging/current_cost���;����+       ��K	����A�+*

logging/current_cost���;	B��+       ��K	���A�+*

logging/current_cost���;����+       ��K	����A�+*

logging/current_cost��;<�{�+       ��K	��A�+*

logging/current_cost�s�;)�6+       ��K	=��A�+*

logging/current_cost`��;���+       ��K	~j��A�,*

logging/current_cost���;3��+       ��K	b���A�,*

logging/current_cost;��;��G�+       ��K	����A�,*

logging/current_cost���;�R9�+       ��K	}���A�,*

logging/current_cost�1�;e!�
+       ��K	}"��A�,*

logging/current_cost�;{*��+       ��K	A[��A�,*

logging/current_cost�j�;S2��+       ��K	ɉ��A�,*

logging/current_cost���;;4�+       ��K	����A�,*

logging/current_cost�E�;^'#�+       ��K	���A�,*

logging/current_cost�8�;7�+       ��K	���A�,*

logging/current_costB!�;\��~+       ��K	�A��A�,*

logging/current_cost!�;ș�+       ��K	Sp��A�,*

logging/current_costU�;~��W+       ��K	����A�,*

logging/current_cost|[�;i0k0+       ��K	����A�,*

logging/current_cost�1�;��*+       ��K	 ���A�,*

logging/current_cost�O�;��e+       ��K	�-���A�,*

logging/current_cost`��;��7+       ��K	�Z���A�,*

logging/current_costY��;�ܩ+       ��K	>����A�,*

logging/current_cost��; ��+       ��K	x����A�,*

logging/current_cost��;n�T�+       ��K	T����A�,*

logging/current_cost�f�;j���+       ��K	���A�,*

logging/current_cost�#�;Y��9+       ��K	�G��A�,*

logging/current_cost��;��+�+       ��K	w��A�,*

logging/current_costpw�;'���+       ��K	����A�,*

logging/current_cost	��;)�U�+       ��K	����A�,*

logging/current_cost$��;B�W�+       ��K	���A�-*

logging/current_cost��;,�+       ��K	�1��A�-*

logging/current_cost���;�h��+       ��K	�`��A�-*

logging/current_cost)��;J�(I+       ��K	����A�-*

logging/current_cost'��;���+       ��K	���A�-*

logging/current_cost5��; �O5+       ��K	���A�-*

logging/current_cost���;@I+       ��K	��A�-*

logging/current_cost���;��{+       ��K	�H��A�-*

logging/current_cost��;*K+       ��K	���A�-*

logging/current_costp��;�
�+       ��K	���A�-*

logging/current_costgh�;��&+       ��K	����A�-*

logging/current_costnl�;�B�+       ��K	���A�-*

logging/current_costUw�;��+       ��K	�9��A�-*

logging/current_cost�0�;�	��+       ��K	�j��A�-*

logging/current_cost��;k5ؖ+       ��K	���A�-*

logging/current_cost)�;�لu+       ��K	s���A�-*

logging/current_costT�;�?3�+       ��K	{���A�-*

logging/current_cost���;b�Ql+       ��K	C!���A�-*

logging/current_cost��;�`W�+       ��K	}Q���A�-*

logging/current_cost ��;�BS�+       ��K	����A�-*

logging/current_costB+�;z��=+       ��K	�����A�-*

logging/current_cost���;󛀯+       ��K	�����A�-*

logging/current_cost{6�;.�5
+       ��K	����A�-*

logging/current_cost`b�;߽�+       ��K	�J���A�-*

logging/current_costu�;���6+       ��K	W���A�-*

logging/current_coste��;��1�+       ��K	�����A�-*

logging/current_costw��;�Z�+       ��K	����A�.*

logging/current_costy��;��Ğ+       ��K	����A�.*

logging/current_cost�g�;��
+       ��K	�I���A�.*

logging/current_cost|G�;�L�+       ��K	����A�.*

logging/current_cost�q�;a�+       ��K	A����A�.*

logging/current_costp�;Og˹+       ��K	;����A�.*

logging/current_cost��;'�56+       ��K	����A�.*

logging/current_costk8�;�_��+       ��K	�@���A�.*

logging/current_cost�[�;�T��+       ��K	�p���A�.*

logging/current_cost|l�;�/��+       ��K	����A�.*

logging/current_cost�C�;eeυ+       ��K	�����A�.*

logging/current_cost�-�;t�%a+       ��K	���A�.*

logging/current_costws�;T�
m+       ��K	�.���A�.*

logging/current_costN��;��i!+       ��K	A\���A�.*

logging/current_costD��;H�;+       ��K	����A�.*

logging/current_cost�u�;�[��+       ��K	*����A�.*

logging/current_cost�j�;�?��+       ��K	�����A�.*

logging/current_cost#�;��l3+       ��K	����A�.*

logging/current_cost�.�;���+       ��K	1M���A�.*

logging/current_cost�l�;u%+       ��K	�{���A�.*

logging/current_cost�k�;yȳk+       ��K	����A�.*

logging/current_cost��;��}+       ��K	�����A�.*

logging/current_costB��;`�8�+       ��K	V���A�.*

logging/current_cost���;��Y�+       ��K	V:���A�.*

logging/current_cost�n�;���+       ��K	j���A�.*

logging/current_cost�g�;�TS�+       ��K	[����A�.*

logging/current_cost��;���+       ��K	����A�/*

logging/current_cost҃�;z��+       ��K	sU���A�/*

logging/current_cost���;Y@N�+       ��K	ɕ���A�/*

logging/current_costD-�;��A+       ��K	����A�/*

logging/current_cost���;���+       ��K	����A�/*

logging/current_cost�;"�Թ+       ��K	$I���A�/*

logging/current_cost@)�;
���+       ��K	3����A�/*

logging/current_cost���;��(�+       ��K	�����A�/*

logging/current_cost ��;�F+       ��K	����A�/*

logging/current_cost|.�;���+       ��K	�'���A�/*

logging/current_cost� �;j�!�+       ��K	Sk���A�/*

logging/current_cost���; ��.+       ��K	S����A�/*

logging/current_cost��;�/�+       ��K	Z����A�/*

logging/current_cost2&�;�Ӗ+       ��K	����A�/*

logging/current_cost���;��+8+       ��K	�@���A�/*

logging/current_cost��;Q�?�+       ��K	v���A�/*

logging/current_costB��;��v+       ��K	Y����A�/*

logging/current_cost���;X��+       ��K	s����A�/*

logging/current_cost��;�-�+       ��K	� ��A�/*

logging/current_cost+��;f��+       ��K	QU ��A�/*

logging/current_costN%�;*f+       ��K	$� ��A�/*

logging/current_cost��;9��+       ��K	�� ��A�/*

logging/current_cost�V�;�7++       ��K	[��A�/*

logging/current_costN��;�*GL+       ��K	�I��A�/*

logging/current_cost4�;gh�T+       ��K	&���A�/*

logging/current_cost���;խ��+       ��K	U���A�0*

logging/current_cost���;���+       ��K	���A�0*

logging/current_costՉ�;����+       ��K	 ���A�0*

logging/current_cost\��;8]��+       ��K	A���A�0*

logging/current_costw��;{J�+       ��K	���A�0*

logging/current_cost`��;�B��+       ��K	La��A�0*

logging/current_cost��;Ow+       ��K	U���A�0*

logging/current_cost��;O��+       ��K	J���A�0*

logging/current_cost�7�; XUc+       ��K	��A�0*

logging/current_cost�^�;북l+       ��K	�U��A�0*

logging/current_costk��;X��c+       ��K	����A�0*

logging/current_costU�;=�#+       ��K	���A�0*

logging/current_cost��;���+       ��K	� ��A�0*

logging/current_cost���;3V�[+       ��K	,5��A�0*

logging/current_cost���;��9�+       ��K	Qe��A�0*

logging/current_cost�%�;}���+       ��K	>���A�0*

logging/current_cost�s�;�蚾+       ��K	����A�0*

logging/current_cost�Z�;8��+       ��K	��A�0*

logging/current_cost���;3�!K+       ��K	�=��A�0*

logging/current_cost4��;-��*+       ��K	tw��A�0*

logging/current_cost��;��x�+       ��K	ɶ��A�0*

logging/current_cost�Y�;�|�+       ��K	U���A�0*

logging/current_cost���;�[�+       ��K	Q7��A�0*

logging/current_costI��;rE+       ��K	�o��A�0*

logging/current_cost���;�Ema+       ��K	���A�0*

logging/current_costL��;���&+       ��K	����A�0*

logging/current_cost�+�;�)+       ��K	s��A�1*

logging/current_costJ�;.U}%+       ��K	'V��A�1*

logging/current_cost�L�;=L+       ��K	����A�1*

logging/current_cost"�;���I+       ��K	���A�1*

logging/current_cost|.�;��+       ��K	�	��A�1*

logging/current_costYT�;f��6+       ��K	�E	��A�1*

logging/current_cost'�;��W+       ��K	O�	��A�1*

logging/current_cost$d�;���c+       ��K	I�	��A�1*

logging/current_cost���;q��+       ��K	?
��A�1*

logging/current_cost0��;W��f+       ��K	�W
��A�1*

logging/current_cost�O�;�q�+       ��K	`�
��A�1*

logging/current_costt��;��w+       ��K	��
��A�1*

logging/current_cost���;*�M+       ��K	��A�1*

logging/current_cost�,�;���+       ��K	�6��A�1*

logging/current_cost�:�;�>�t+       ��K	�q��A�1*

logging/current_costgK�;��d�+       ��K	6���A�1*

logging/current_cost���;n�|F+       ��K	>���A�1*

logging/current_cost���;�zI�+       ��K	/��A�1*

logging/current_costw��;j��+       ��K	�f��A�1*

logging/current_cost�p�;	��+       ��K	����A�1*

logging/current_cost���;�6��+       ��K	����A�1*

logging/current_cost`��;��Z+       ��K	���A�1*

logging/current_cost���;9[5y+       ��K	�<��A�1*

logging/current_cost��;���+       ��K	�m��A�1*

logging/current_cost���;[��{+       ��K	���A�1*

logging/current_cost��;;�%x+       ��K	����A�2*

logging/current_cost�q�;!=�+       ��K	��A�2*

logging/current_costº�;I_V�+       ��K	�S��A�2*

logging/current_cost���;bF%<+       ��K	����A�2*

logging/current_cost�;�H��+       ��K	���A�2*

logging/current_cost��;]���+       ��K	H���A�2*

logging/current_cost'�;��2�+       ��K	e!��A�2*

logging/current_cost��;�) +       ��K	V��A�2*

logging/current_cost���;�^Z+       ��K	͈��A�2*

logging/current_costT��;9��/+       ��K	^���A�2*

logging/current_cost��;��T+       ��K	����A�2*

logging/current_cost���;���+       ��K	m8��A�2*

logging/current_cost4��;���	+       ��K	���A�2*

logging/current_cost��;��J�+       ��K	����A�2*

logging/current_cost�!�;5�_}+       ��K	����A�2*

logging/current_cost�,�;2(C�+       ��K	��A�2*

logging/current_cost���;��y+       ��K	�M��A�2*

logging/current_cost��;���+       ��K	����A�2*

logging/current_cost�>�;��Ba+       ��K	<���A�2*

logging/current_cost'�;��+       ��K	���A�2*

logging/current_cost��;��yE+       ��K	���A�2*

logging/current_costt:�;���+       ��K	�M��A�2*

logging/current_cost�m�;��Z+       ��K	{���A�2*

logging/current_cost�C�;d�R+       ��K	����A�2*

logging/current_cost�;G�j+       ��K	����A�2*

logging/current_cost�_�;!�.�+       ��K	���A�2*

logging/current_cost��;�L0+       ��K	Q��A�3*

logging/current_cost�a�;��^�+       ��K	���A�3*

logging/current_cost�L�;\��z+       ��K	���A�3*

logging/current_cost��;��B<+       ��K	&���A�3*

logging/current_cost|,�;�V+       ��K	�*��A�3*

logging/current_costU��;��1�+       ��K	�e��A�3*

logging/current_cost�F�;��T+       ��K	����A�3*

logging/current_cost�[�;�|�+       ��K	����A�3*

logging/current_costK)�;���+       ��K	@��A�3*

logging/current_cost���;;���+       ��K	"@��A�3*

logging/current_cost��;)ݞ�+       ��K	�}��A�3*

logging/current_cost���;l�޳+       ��K	����A�3*

logging/current_costu��;��:x+       ��K	����A�3*

logging/current_costY{�;��m�+       ��K	9,��A�3*

logging/current_cost�d�;Pߐ	+       ��K	�g��A�3*

logging/current_cost��;�X+       ��K	h���A�3*

logging/current_cost��;Bp[+       ��K	����A�3*

logging/current_cost;��;�e��+       ��K	l+��A�3*

logging/current_cost�>�;��IS+       ��K	>c��A�3*

logging/current_cost7�;{�O+       ��K	Q���A�3*

logging/current_costtr�;�\�+       ��K	^���A�3*

logging/current_cost�+�;�+       ��K	���A�3*

logging/current_cost)�;΁�+       ��K	�Y��A�3*

logging/current_cost&�;z�1+       ��K	���A�3*

logging/current_costd�;�>�+       ��K	����A�3*

logging/current_cost7�;��x-+       ��K	���A�3*

logging/current_cost'�;%^�A+       ��K	^N��A�4*

logging/current_cost���;�h+       ��K	h���A�4*

logging/current_costu=�;`�K+       ��K	-���A�4*

logging/current_cost���;�'�+       ��K	����A�4*

logging/current_cost���;�U��+       ��K	2*��A�4*

logging/current_cost�;�Y��+       ��K	�[��A�4*

logging/current_costd��;��Bu+       ��K	���A�4*

logging/current_cost���;K�n�+       ��K	����A�4*

logging/current_cost���;�G�+       ��K	r��A�4*

logging/current_costU��;X�+       ��K	�B��A�4*

logging/current_cost9��;0�9+       ��K	܇��A�4*

logging/current_costk��;)F�+       ��K	���A�4*

logging/current_costb��;���+       ��K	�F��A�4*

logging/current_cost
�;���+       ��K	�y��A�4*

logging/current_cost$��;$��+       ��K	ɳ��A�4*

logging/current_cost5��;��q+       ��K	o���A�4*

logging/current_cost��;�QL+       ��K	�*��A�4*

logging/current_cost��;;�$+       ��K	^��A�4*

logging/current_cost$��;x�qK+       ��K	V���A�4*

logging/current_costp0�;
c+       ��K	���A�4*

logging/current_cost��;;64+       ��K	#��A�4*

logging/current_cost���;�]�+       ��K	H��A�4*

logging/current_costY�;��B�+       ��K	�~��A�4*

logging/current_cost ��;�	�+       ��K	����A�4*

logging/current_cost���;-��+       ��K	����A�4*

logging/current_cost@*�;�c%�+       ��K	z(��A�5*

logging/current_cost5��;�mB�+       ��K	im��A�5*

logging/current_cost'��;��o+       ��K	����A�5*

logging/current_cost��;�'+       ��K	����A�5*

logging/current_cost���;K��+       ��K	�4 ��A�5*

logging/current_cost��;"���+       ��K	�j ��A�5*

logging/current_cost�l�;#�r+       ��K	ڡ ��A�5*

logging/current_costZ�;�l/i+       ��K	� ��A�5*

logging/current_cost��;��+       ��K	q!��A�5*

logging/current_costٕ�;���R+       ��K	=R!��A�5*

logging/current_cost�t�;�[�+       ��K	�!��A�5*

logging/current_costĝ�;7}M+       ��K	��!��A�5*

logging/current_cost���;�i+       ��K	]�!��A�5*

logging/current_costI��;���X+       ��K	�&"��A�5*

logging/current_cost'��;�e҂+       ��K	�d"��A�5*

logging/current_cost��;]2��+       ��K	^�"��A�5*

logging/current_costW��;�ÕA+       ��K	�"��A�5*

logging/current_costy��;N�:G+       ��K	W�"��A�5*

logging/current_cost�m�;�0�+       ��K	�9#��A�5*

logging/current_costbk�;S�}�+       ��K	�v#��A�5*

logging/current_cost�=�;v~�+       ��K	�#��A�5*

logging/current_cost�[�;�J��+       ��K	��#��A�5*

logging/current_cost�7�;}{�+       ��K	�$��A�5*

logging/current_cost'
�;W"�+       ��K	�K$��A�5*

logging/current_cost�3�;�_#+       ��K	I�$��A�5*

logging/current_cost #�;+�_�+       ��K	��$��A�5*

logging/current_cost�.�;�FAP+       ��K	f�$��A�6*

logging/current_cost���;�Q(+       ��K	�(%��A�6*

logging/current_cost$��;�w�+       ��K	�c%��A�6*

logging/current_cost��;	D��+       ��K	�%��A�6*

logging/current_costu�;DX��+       ��K	��%��A�6*

logging/current_cost���;˵��+       ��K	`&��A�6*

logging/current_costl��;qBR+       ��K	�8&��A�6*

logging/current_costt�;���+       ��K	�m&��A�6*

logging/current_cost��;�+       ��K	#�&��A�6*

logging/current_cost���;�H�>+       ��K	��&��A�6*

logging/current_cost���;��W+       ��K	�'��A�6*

logging/current_cost�I�;�xW�+       ��K	�T'��A�6*

logging/current_costҧ�;oOS+       ��K	�'��A�6*

logging/current_cost�{�;Г�5+       ��K	��'��A�6*

logging/current_cost���;Hs�!+       ��K	�(��A�6*

logging/current_cost�t�;{�k+       ��K	�K(��A�6*

logging/current_cost��;'�5+       ��K	�(��A�6*

logging/current_costB��;�.�+       ��K	��(��A�6*

logging/current_cost���;�[��+       ��K	��(��A�6*

logging/current_cost0�;AU"+       ��K	�7)��A�6*

logging/current_cost�>�;�'+       ��K	Z�)��A�6*

logging/current_cost��;|ͦ�+       ��K	��)��A�6*

logging/current_costKk�;�8�F+       ��K	T*��A�6*

logging/current_cost��;�jRe+       ��K	DM*��A�6*

logging/current_cost�M�;�8+       ��K	��*��A�6*

logging/current_cost���;Q��B+       ��K	��*��A�7*

logging/current_costn�;L��+       ��K	2+��A�7*

logging/current_cost[;�;�j+       ��K	
F+��A�7*

logging/current_cost��;j��+       ��K	�u+��A�7*

logging/current_cost���;8ͩ�+       ��K	p�+��A�7*

logging/current_cost���;�Pt�+       ��K	!�+��A�7*

logging/current_costuq�;���+       ��K	�,��A�7*

logging/current_cost�L�;�Ά�+       ��K	 Q,��A�7*

logging/current_costK[�;T,+       ��K	K�,��A�7*

logging/current_costpN�;���`+       ��K	a�,��A�7*

logging/current_cost��;�e�+       ��K	'�,��A�7*

logging/current_costI�;����+       ��K	�--��A�7*

logging/current_cost��;��+       ��K	�`-��A�7*

logging/current_cost��;��+       ��K	�-��A�7*

logging/current_cost<}�;̮��+       ��K	��-��A�7*

logging/current_cost$��;t��+       ��K	�.��A�7*

logging/current_cost���;�80<+       ��K	�N.��A�7*

logging/current_cost���;bXT�+       ��K	i�.��A�7*

logging/current_cost�$�;X5+       ��K	��.��A�7*

logging/current_cost,X�;��:+       ��K	�/��A�7*

logging/current_costB�;��.+       ��K	I`/��A�7*

logging/current_cost���;�0�d+       ��K	��/��A�7*

logging/current_cost�;��?�+       ��K	~�/��A�7*

logging/current_costd��;i*@+       ��K	n�/��A�7*

logging/current_costU��;���,+       ��K	�=0��A�7*

logging/current_cost�[�;I�;+       ��K	[u0��A�7*

logging/current_cost�v�;�T/D+       ��K	l�0��A�8*

logging/current_costΪ�;� p�+       ��K	L�0��A�8*

logging/current_costn��;%X�+       ��K	�91��A�8*

logging/current_costU��;���l+       ��K	/r1��A�8*

logging/current_costR�;O*�o+       ��K	�1��A�8*

logging/current_cost]�;b��V+       ��K	��1��A�8*

logging/current_costY�;���+       ��K	�2��A�8*

logging/current_cost��;����+       ��K	8F2��A�8*

logging/current_costB�;)���+       ��K	x�2��A�8*

logging/current_cost�@�;f7&+       ��K	��2��A�8*

logging/current_cost�#�;!&a)+       ��K	�2��A�8*

logging/current_cost�.�;<��+       ��K	�>3��A�8*

logging/current_costd#�;+�A�+       ��K	�y3��A�8*

logging/current_cost>�;<�+       ��K	��3��A�8*

logging/current_cost=�;��fB+       ��K	��3��A�8*

logging/current_cost7��;h�'+       ��K	�4��A�8*

logging/current_costt=�;�k7.+       ��K	�Q4��A�8*

logging/current_cost�Y�;n���+       ��K	?�4��A�8*

logging/current_cost�m�;��+       ��K	��4��A�8*

logging/current_cost	��;j��c+       ��K	�5��A�8*

logging/current_costT{�;f�:+       ��K	K5��A�8*

logging/current_cost��;��ǡ+       ��K	7�5��A�8*

logging/current_costN��;Ak��+       ��K	��5��A�8*

logging/current_cost� �;�\O�+       ��K	��5��A�8*

logging/current_cost���;��a+       ��K	k6��A�8*

logging/current_cost2��;�b�m+       ��K	*J6��A�8*

logging/current_cost;/�;i�?�+       ��K	П6��A�9*

logging/current_cost;x�;J.+       ��K	$�6��A�9*

logging/current_costa�;2���+       ��K	�7��A�9*

logging/current_cost@[�;vD7+       ��K	�<7��A�9*

logging/current_cost�f�;��4h+       ��K	}7��A�9*

logging/current_cost ��;�|��+       ��K	߽7��A�9*

logging/current_cost$��;O�t�+       ��K	.�7��A�9*

logging/current_cost\��;u��+       ��K	� 8��A�9*

logging/current_cost4��;�B;+       ��K	�P8��A�9*

logging/current_costnH�;�t�+       ��K	��8��A�9*

logging/current_cost�3�;���+       ��K	�8��A�9*

logging/current_cost D�;K��+       ��K	��8��A�9*

logging/current_cost���;X~A+       ��K	a!9��A�9*

logging/current_cost\�;(�8�+       ��K	�N9��A�9*

logging/current_cost\�;�D�+       ��K	�9��A�9*

logging/current_cost9��;~Rѳ+       ��K	w�9��A�9*

logging/current_cost#�;���+       ��K	_�9��A�9*

logging/current_cost�G�;}���+       ��K	*!:��A�9*

logging/current_cost��;�n&+       ��K	%[:��A�9*

logging/current_cost$�;֔�+       ��K	o�:��A�9*

logging/current_cost��;�=�+       ��K	��:��A�9*

logging/current_cost�r�;C��+       ��K	�;��A�9*

logging/current_cost�H�;�UY+       ��K	�G;��A�9*

logging/current_costN
�;/j�+       ��K	-�;��A�9*

logging/current_cost ��;sP��+       ��K	��;��A�9*

logging/current_costY��;���+       ��K	�"<��A�:*

logging/current_cost���;�Z�+       ��K	�`<��A�:*

logging/current_cost���;*��+       ��K	��<��A�:*

logging/current_costu��;��+       ��K	�<��A�:*

logging/current_cost�$�;.��V+       ��K	�#=��A�:*

logging/current_cost�@�;M�:h+       ��K	�V=��A�:*

logging/current_cost�G�;�]N�+       ��K	Q�=��A�:*

logging/current_cost�@�;ks��+       ��K	��=��A�:*

logging/current_costb�;g +       ��K	d>��A�:*

logging/current_cost��;�f:�+       ��K	X>��A�:*

logging/current_cost���;B�yf+       ��K	��>��A�:*

logging/current_cost��;����+       ��K	�>��A�:*

logging/current_costĽ�;u+.g+       ��K	��>��A�:*

logging/current_cost��;��c+       ��K	_(?��A�:*

logging/current_cost��;YŜ�+       ��K	�c?��A�:*

logging/current_costZ�;���G+       ��K	��?��A�:*

logging/current_costy��;q"k$+       ��K	��?��A�:*

logging/current_cost`�;��6�+       ��K	�@��A�:*

logging/current_cost��;N�A�+       ��K	SJ@��A�:*

logging/current_cost�X�;ޚ�0+       ��K	h�@��A�:*

logging/current_costխ�;:���+       ��K	j�@��A�:*

logging/current_costg��;��}3+       ��K	�	A��A�:*

logging/current_costɗ�;Ci*+       ��K	�CA��A�:*

logging/current_cost]�;�2VH+       ��K	zA��A�:*

logging/current_cost\~�;��+       ��K	ɫA��A�:*

logging/current_cost\��;��b�+       ��K	��A��A�:*

logging/current_cost���;k?�+       ��K	1B��A�;*

logging/current_cost�/�;��$+       ��K	lWB��A�;*

logging/current_cost<�;Y���+       ��K	őB��A�;*

logging/current_cost���;�s`s+       ��K	��B��A�;*

logging/current_cost���;���+       ��K	 QC��A�;*

logging/current_cost���;��!X+       ��K	��C��A�;*

logging/current_cost��;���N+       ��K	7�C��A�;*

logging/current_costt��;n$�+       ��K	5YD��A�;*

logging/current_cost���;���+       ��K	?�D��A�;*

logging/current_cost��;Z�*�+       ��K	�E��A�;*

logging/current_cost[��;5�+       ��K	�VE��A�;*

logging/current_costKd�;�9�R+       ��K	Q�E��A�;*

logging/current_cost^�;ʼ�f+       ��K	��E��A�;*

logging/current_cost"��;� ��+       ��K	�F��A�;*

logging/current_cost���;	$�!+       ��K	�>F��A�;*

logging/current_cost1�;�)��+       ��K	osF��A�;*

logging/current_cost���;���+       ��K	n�F��A�;*

logging/current_cost�E�;Ua�+       ��K	��F��A�;*

logging/current_cost>C�;�y+       ��K	�G��A�;*

logging/current_cost�:�;}��+       ��K	QG��A�;*

logging/current_cost�;R���+       ��K	Q�G��A�;*

logging/current_cost t�;�7(�+       ��K	�G��A�;*

logging/current_cost���;�U:�+       ��K	�H��A�;*

logging/current_costT@�;ʰ�z+       ��K	�EH��A�;*

logging/current_costA�;)^�3+       ��K	�yH��A�;*

logging/current_cost�9�;�-@�+       ��K	�H��A�<*

logging/current_cost���;+�yT+       ��K	�I��A�<*

logging/current_cost�	�;�?׳+       ��K	�JI��A�<*

logging/current_cost2E�;,�	+       ��K	9~I��A�<*

logging/current_cost�<�;�g�=+       ��K	�I��A�<*

logging/current_cost�|�;U���+       ��K	�J��A�<*

logging/current_cost��;���+       ��K	�\J��A�<*

logging/current_cost�7�;@1�+       ��K	*�J��A�<*

logging/current_cost,G�;�x+       ��K	�J��A�<*

logging/current_cost }�;��w+       ��K	�J��A�<*

logging/current_cost>��;�?��+       ��K	�0K��A�<*

logging/current_cost�o�;�Y��+       ��K	-cK��A�<*

logging/current_cost?�;����+       ��K	6�K��A�<*

logging/current_cost�$�;�٤�+       ��K	��K��A�<*

logging/current_cost`��;u̶X+       ��K	�L��A�<*

logging/current_costt��;��+       ��K	�GL��A�<*

logging/current_cost.(�;>��+       ��K	>�L��A�<*

logging/current_cost٫�;�n�+       ��K	z�L��A�<*

logging/current_cost���;T!�+       ��K	�M��A�<*

logging/current_cost�&�;� +       ��K	&�M��A�<*

logging/current_cost���;�G��+       ��K	(�M��A�<*

logging/current_cost"��;���+       ��K	K�M��A�<*

logging/current_cost`��;�8�x+       ��K	�#N��A�<*

logging/current_cost���;T���+       ��K	/ZN��A�<*

logging/current_cost̎�;��+       ��K	�N��A�<*

logging/current_cost���;v�-�+       ��K	��N��A�<*

logging/current_cost>��;��=+       ��K	��N��A�=*

logging/current_cost���;w/��+       ��K	�3O��A�=*

logging/current_costT��;�d�+       ��K	�lO��A�=*

logging/current_cost�C�;�I��+       ��K	��O��A�=*

logging/current_cost5��;>5��+       ��K	��O��A�=*

logging/current_cost���; ��+       ��K	�P��A�=*

logging/current_cost ��;T��+       ��K	�AP��A�=*

logging/current_cost�[�;�� �+       ��K	�wP��A�=*

logging/current_cost� �;X��g+       ��K	e�P��A�=*

logging/current_cost`2�;�ST�+       ��K	��P��A�=*

logging/current_costg%�;R�m�+       ��K	�Q��A�=*

logging/current_costg�;U��+       ��K	�AQ��A�=*

logging/current_cost�1�;6�I+       ��K	�yQ��A�=*

logging/current_cost���;��h+       ��K	��Q��A�=*

logging/current_cost�;=t��+       ��K	K�Q��A�=*

logging/current_cost;"�;~V�$+       ��K	�R��A�=*

logging/current_costgo�;��%b+       ��K	zRR��A�=*

logging/current_costIH�;$p�+       ��K	3�R��A�=*

logging/current_cost�y�;A?�+       ��K	$�R��A�=*

logging/current_cost�0�;'F�+       ��K		S��A�=*

logging/current_cost0��;;},t+       ��K	1VS��A�=*

logging/current_cost���;
�7�+       ��K	�S��A�=*

logging/current_cost���;�3_�+       ��K	J�S��A�=*

logging/current_cost#�;�▪+       ��K	�
T��A�=*

logging/current_cost���;��D�+       ��K	-GT��A�=*

logging/current_cost�)�;	�q�+       ��K	��T��A�=*

logging/current_costi��;iK��+       ��K	�T��A�>*

logging/current_cost2�;��+�+       ��K	��T��A�>*

logging/current_costRj�;ċ+       ��K	X0U��A�>*

logging/current_cost^�;� #+       ��K	ymU��A�>*

logging/current_cost�9�;8m�	+       ��K	+�U��A�>*

logging/current_cost��;����+       ��K	K�U��A�>*

logging/current_costl��;��Ya+       ��K	�V��A�>*

logging/current_cost2�;�*�W+       ��K	+?V��A�>*

logging/current_cost�"�;;�+       ��K	�V��A�>*

logging/current_cost;^�;Q��r+       ��K	��V��A�>*

logging/current_cost���;r��P+       ��K	z�V��A�>*

logging/current_cost["�;c�C
+       ��K	4'W��A�>*

logging/current_cost�)�;�UK�+       ��K	�[W��A�>*

logging/current_cost��;�$�5+       ��K	4�W��A�>*

logging/current_cost�Z�;���_+       ��K	��W��A�>*

logging/current_cost���;0os+       ��K	�X��A�>*

logging/current_costY�;���+       ��K	IX��A�>*

logging/current_costN�;}�*+       ��K	��X��A�>*

logging/current_cost<}�;w׆+       ��K	E�X��A�>*

logging/current_cost�Z�;����+       ��K	�0Y��A�>*

logging/current_cost���;�$�+       ��K	�hY��A�>*

logging/current_cost�L�;;��+       ��K	��Y��A�>*

logging/current_cost�C�;Z8�+       ��K	��Y��A�>*

logging/current_cost0��;-Ew\+       ��K	>Z��A�>*

logging/current_cost���;'�K+       ��K	�IZ��A�>*

logging/current_costj�;sǙ	+       ��K	��Z��A�?*

logging/current_cost�\�;@S�:+       ��K	.�Z��A�?*

logging/current_cost���;Ϯd+       ��K	��Z��A�?*

logging/current_cost���;�T�+       ��K	8?[��A�?*

logging/current_costk��;5�;+       ��K	7u[��A�?*

logging/current_cost+A�;Hu��+       ��K	�[��A�?*

logging/current_cost��;�b%+       ��K	[�[��A�?*

logging/current_costK/�;3�Ϣ+       ��K	_\��A�?*

logging/current_cost\L�;w"/�+       ��K	M\��A�?*

logging/current_cost���;�R;+       ��K	a�\��A�?*

logging/current_cost���;�_��+       ��K	��\��A�?*

logging/current_costN��;�@O+       ��K	��\��A�?*

logging/current_cost\��;����+       ��K	�5]��A�?*

logging/current_costw��;��+       ��K	�j]��A�?*

logging/current_cost�2�;�A�x+       ��K	 �]��A�?*

logging/current_costb��;C���+       ��K	c�]��A�?*

logging/current_cost@�;�h+       ��K	�^��A�?*

logging/current_cost�*�;��!/+       ��K	�@^��A�?*

logging/current_cost�`�;s�K}+       ��K	q^��A�?*

logging/current_cost�o�;5�+       ��K	��^��A�?*

logging/current_cost$B�;"j��+       ��K	a�^��A�?*

logging/current_cost�U�;����+       ��K	u�^��A�?*

logging/current_cost!�;u� 9+       ��K	T2_��A�?*

logging/current_cost'�;C+       ��K	�b_��A�?*

logging/current_cost��;�@�+       ��K	�_��A�?*

logging/current_cost$��;��+       ��K	!�_��A�?*

logging/current_cost��;���F+       ��K	<`��A�@*

logging/current_cost5��;��0+       ��K	�>`��A�@*

logging/current_cost���;;G�+       ��K	�o`��A�@*

logging/current_cost��;!X!+       ��K	Y�`��A�@*

logging/current_costY�;�XP�+       ��K	@�`��A�@*

logging/current_costB3�;�J��+       ��K	��`��A�@*

logging/current_costx�;lG��+       ��K	q6a��A�@*

logging/current_cost@5�;�J&+       ��K	:ia��A�@*

logging/current_cost�n�;�Vn+       ��K	p�a��A�@*

logging/current_costn��;̼�.+       ��K	g�a��A�@*

logging/current_costd��;���i+       ��K	B�a��A�@*

logging/current_costu�;��ޯ+       ��K	�/b��A�@*

logging/current_cost�S�;�x�+       ��K	gb��A�@*

logging/current_cost���;�xٙ+       ��K	"�b��A�@*

logging/current_cost{��;*�I1+       ��K	*�b��A�@*

logging/current_costk4�;m�!~+       ��K	��b��A�@*

logging/current_cost���;s �+       ��K	�/c��A�@*

logging/current_costP�;vD0�+       ��K	�ac��A�@*

logging/current_cost���;���+       ��K		�c��A�@*

logging/current_costT�;��Yu+       ��K	!�c��A�@*

logging/current_cost�X�;K���+       ��K	��c��A�@*

logging/current_cost7��;���+       ��K	�0d��A�@*

logging/current_cost@�;]%��+       ��K	hnd��A�@*

logging/current_cost@�;dN*�+       ��K	��d��A�@*

logging/current_costŞ�;[��+       ��K	�d��A�@*

logging/current_cost~�;{h*z+       ��K	�e��A�A*

logging/current_coste��;v�c/+       ��K	k<e��A�A*

logging/current_cost���;~#0+       ��K	�ve��A�A*

logging/current_cost���;k�{�+       ��K	��e��A�A*

logging/current_cost�0�;[��B+       ��K	�e��A�A*

logging/current_cost�w�;�I�+       ��K	�f��A�A*

logging/current_cost��;_�"*+       ��K	&Lf��A�A*

logging/current_cost$N�;�\�+       ��K	T�f��A�A*

logging/current_cost��;A��+       ��K	ܵf��A�A*

logging/current_cost�"�;��ܻ+       ��K	�f��A�A*

logging/current_cost ��;'�+       ��K	_(g��A�A*

logging/current_cost���;.��+       ��K	hg��A�A*

logging/current_cost�I�;�0_�+       ��K	(�g��A�A*

logging/current_cost�;3#]�+       ��K	?�g��A�A*

logging/current_cost��;	*�+       ��K	)4h��A�A*

logging/current_cost���;�7+       ��K	�ih��A�A*

logging/current_costR��;I�e1+       ��K	)�h��A�A*

logging/current_costD2�;a�8�+       ��K	��h��A�A*

logging/current_cost��;Z���+       ��K	�i��A�A*

logging/current_cost5�;�y�+       ��K	qii��A�A*

logging/current_costR�;P�O+       ��K	�i��A�A*

logging/current_cost2T�;v2�+       ��K	�j��A�A*

logging/current_costk��;�q+       ��K	/Hj��A�A*

logging/current_cost'��;Z�H+       ��K	
�j��A�A*

logging/current_cost|�;�gX+       ��K	��j��A�A*

logging/current_cost\^�;`���+       ��K	^k��A�A*

logging/current_cost)��;�
+       ��K	�Ik��A�B*

logging/current_costL��;�?+       ��K	:�k��A�B*

logging/current_costK��;���m+       ��K	_�k��A�B*

logging/current_cost��;8	�+       ��K	�	l��A�B*

logging/current_costD �;���+       ��K	Il��A�B*

logging/current_costP��;Y��+       ��K	X�l��A�B*

logging/current_costG0�;��%�+       ��K	x�l��A�B*

logging/current_cost�C�;��+       ��K	��l��A�B*

logging/current_cost;��;��+       ��K	42m��A�B*

logging/current_costu�;bQ�+       ��K	�xm��A�B*

logging/current_cost<c�;D�+       ��K	�m��A�B*

logging/current_cost�2�;.c+       ��K	z�m��A�B*

logging/current_costtZ�;
�C+       ��K	'n��A�B*

logging/current_cost%��;�_U+       ��K	�[n��A�B*

logging/current_costR��;���9+       ��K	�n��A�B*

logging/current_cost���;s��+       ��K	:�n��A�B*

logging/current_cost`?�;X�+       ��K	��n��A�B*

logging/current_costk�;p��N+       ��K	� o��A�B*

logging/current_costŬ�;�aص+       ��K	�\o��A�B*

logging/current_cost�@�;&_��+       ��K	=�o��A�B*

logging/current_cost��;���3+       ��K	O�o��A�B*

logging/current_cost�5�;��̣+       ��K	W�o��A�B*

logging/current_cost�8�;�݁+       ��K	9#p��A�B*

logging/current_cost>��;<���+       ��K	8Rp��A�B*

logging/current_cost@��;決�+       ��K	��p��A�B*

logging/current_cost���;7i)�+       ��K	[�p��A�B*

logging/current_cost,�;�]3c+       ��K	�p��A�C*

logging/current_cost\��;��+       ��K	P#q��A�C*

logging/current_cost�4�;�(
�+       ��K	�Vq��A�C*

logging/current_cost0��;�� �+       ��K	шq��A�C*

logging/current_costU��;�[5�+       ��K	�q��A�C*

logging/current_costI"�;Ṭ+       ��K	��q��A�C*

logging/current_cost���;)/9�+       ��K	g]r��A�C*

logging/current_cost��;4q۳+       ��K	�r��A�C*

logging/current_cost���;��z�+       ��K	��r��A�C*

logging/current_cost��;�n�K+       ��K	d�r��A�C*

logging/current_cost4�;[=d+       ��K	:s��A�C*

logging/current_cost�&�;>w�+       ��K	L{s��A�C*

logging/current_costw�;���+       ��K	X�s��A�C*

logging/current_cost��;a+       ��K	u�s��A�C*

logging/current_cost�A�;]�+       ��K	,t��A�C*

logging/current_costY��;�a�+       ��K	Z_t��A�C*

logging/current_cost���;*H�+       ��K	Ξt��A�C*

logging/current_cost��;A��+       ��K	'�t��A�C*

logging/current_cost ^�;0� {+       ��K	-u��A�C*

logging/current_cost���;�1�+       ��K	W=u��A�C*

logging/current_costG��;O@n+       ��K	 wu��A�C*

logging/current_costy��;E�x�+       ��K	��u��A�C*

logging/current_cost�A�;�2��+       ��K	��u��A�C*

logging/current_costk>�;��^�+       ��K	�v��A�C*

logging/current_cost���;Ք�K+       ��K	c;v��A�C*

logging/current_cost�{�;\۪V+       ��K	lv��A�D*

logging/current_costI��;,���+       ��K	�v��A�D*

logging/current_cost)�;Bܠs+       ��K	��v��A�D*

logging/current_cost���;B�*�+       ��K	��v��A�D*

logging/current_cost���;�Ȼ+       ��K	,w��A�D*

logging/current_cost���;���+       ��K	�\w��A�D*

logging/current_cost�u�;p+       ��K	�w��A�D*

logging/current_costE�; ޙ+       ��K	��w��A�D*

logging/current_costU2�;=\�;+       ��K	t�w��A�D*

logging/current_cost���;�Z�w+       ��K	�x��A�D*

logging/current_cost��;�R�+       ��K	MQx��A�D*

logging/current_costU��;�o�t+       ��K	��x��A�D*

logging/current_cost9u�;����+       ��K	��x��A�D*

logging/current_cost@��;��}�+       ��K	��x��A�D*

logging/current_cost���;�s�i+       ��K	Xy��A�D*

logging/current_costg��;��+       ��K	�Cy��A�D*

logging/current_cost6�;b�� +       ��K	Vty��A�D*

logging/current_cost2 �;?9u^+       ��K	_�y��A�D*

logging/current_cost���;^4�m+       ��K	@�y��A�D*

logging/current_cost���;���+       ��K	�z��A�D*

logging/current_costp�;����+       ��K	�1z��A�D*

logging/current_cost��;����+       ��K	^z��A�D*

logging/current_cost���;W�ӯ+       ��K	9�z��A�D*

logging/current_cost@��;v��+       ��K	=�z��A�D*

logging/current_cost5��;�uO�+       ��K	_�z��A�D*

logging/current_cost�a�;��/+       ��K	�{��A�D*

logging/current_cost�k�;��g~+       ��K	�K{��A�E*

logging/current_cost��;��V+       ��K	$�{��A�E*

logging/current_cost��;6��<+       ��K	K�{��A�E*

logging/current_cost$�;-=��+       ��K	z0|��A�E*

logging/current_cost�F�;G
�+       ��K	bm|��A�E*

logging/current_cost��;��k+       ��K	�|��A�E*

logging/current_cost���;��K�+       ��K	r�|��A�E*

logging/current_costޜ�;�s+       ��K	�.}��A�E*

logging/current_cost`��;gjg+       ��K	s}��A�E*

logging/current_cost���;!�:�+       ��K	Ƹ}��A�E*

logging/current_costU�;��,�+       ��K	�,~��A�E*

logging/current_cost��;�c+       ��K	�t~��A�E*

logging/current_cost>��;�&2o+       ��K	�~��A�E*

logging/current_cost^��;�A��+       ��K	Z�~��A�E*

logging/current_costu��;�aW+       ��K	��A�E*

logging/current_cost���;	�ܘ+       ��K	oN��A�E*

logging/current_cost���;[��+       ��K	����A�E*

logging/current_cost"x�;+0(�+       ��K	h���A�E*

logging/current_cost�5�;r���+       ��K	����A�E*

logging/current_cost ��;vޕ�+       ��K	� ���A�E*

logging/current_costL~�;v���+       ��K	�{���A�E*

logging/current_cost�c�;�7{�+       ��K	�����A�E*

logging/current_cost�2�;	���+       ��K	瀇�A�E*

logging/current_cost�f�;��>S+       ��K	;���A�E*

logging/current_costE$�;Ҹ^?+       ��K	�W���A�E*

logging/current_cost�w�;glM�+       ��K	'����A�F*

logging/current_cost�;�O�+       ��K	�΁��A�F*

logging/current_cost���;�#�+       ��K	v
���A�F*

logging/current_cost�6�;�#�+       ��K	/H���A�F*

logging/current_cost���;dH:+       ��K	I����A�F*

logging/current_cost@��;�z+       ��K	�Ԃ��A�F*

logging/current_costg�;cY��+       ��K	;���A�F*

logging/current_cost��;��g�+       ��K	|F���A�F*

logging/current_costB��;'��I+       ��K	����A�F*

logging/current_costD	�;�o��+       ��K	�����A�F*

logging/current_cost���;�\��+       ��K	5惇�A�F*

logging/current_cost��;��Ta+       ��K	�&���A�F*

logging/current_cost�9�;�'�+       ��K	�b���A�F*

logging/current_cost�;��+       ��K	�����A�F*

logging/current_costw��;d?+       ��K	�Ą��A�F*

logging/current_costN�;I�B�+       ��K	@����A�F*

logging/current_costu��;����+       ��K	-'���A�F*

logging/current_cost'2�;��e=+       ��K	�Y���A�F*

logging/current_cost� �;��&�+       ��K	�����A�F*

logging/current_cost���;���+       ��K	�����A�F*

logging/current_cost6�;u��+       ��K	�兇�A�F*

logging/current_cost���;=焱+       ��K	����A�F*

logging/current_cost��;I�+       ��K	�B���A�F*

logging/current_cost�q�;ز�+       ��K		x���A�F*

logging/current_costA�;�#�m+       ��K	����A�F*

logging/current_cost���;p�S�+       ��K	�Ն��A�F*

logging/current_cost��;t�*4+       ��K	����A�G*

logging/current_cost��;\��+       ��K	55���A�G*

logging/current_cost��;�ޅ+       ��K	b���A�G*

logging/current_cost	��;�c+       ��K	c����A�G*

logging/current_cost�&�;�úr+       ��K	LÇ��A�G*

logging/current_cost���;��4�+       ��K	���A�G*

logging/current_costGE�;5���+       ��K	� ���A�G*

logging/current_cost�P�;�a��+       ��K	�M���A�G*

logging/current_cost��;�C��+       ��K	L{���A�G*

logging/current_cost���;n��X+       ��K	'����A�G*

logging/current_cost��;鱶b+       ��K	�ሇ�A�G*

logging/current_cost���;���p+       ��K	����A�G*

logging/current_cost�>�;���+       ��K	2<���A�G*

logging/current_costN��;�a*Y+       ��K	�k���A�G*

logging/current_cost�R�;}�0[+       ��K	�����A�G*

logging/current_cost���;v�
+       ��K	"͉��A�G*

logging/current_cost.2�;?g=w+       ��K	�����A�G*

logging/current_costE��;��l�+       ��K	t(���A�G*

logging/current_cost���;�;��+       ��K	�T���A�G*

logging/current_costrV�;��F	+       ��K	u����A�G*

logging/current_cost;��;6m�D+       ��K	����A�G*

logging/current_cost���;=��+       ��K	銇�A�G*

logging/current_cost��;��ԩ+       ��K	����A�G*

logging/current_cost���;	HW�+       ��K	pG���A�G*

logging/current_cost�;
̨~+       ��K	*u���A�G*

logging/current_cost���;ٚ��+       ��K	s����A�G*

logging/current_costu��;Wm|+       ��K	bы��A�H*

logging/current_cost@x�;Tt^+       ��K	�����A�H*

logging/current_cost,��;W���+       ��K	�,���A�H*

logging/current_cost��;k|�+       ��K	�Y���A�H*

logging/current_cost���;5��+       ��K	ڊ���A�H*

logging/current_cost5D�;悝�+       ��K	u����A�H*

logging/current_costE��;��j+       ��K	�ꌇ�A�H*

logging/current_costU��;�@��+       ��K	����A�H*

logging/current_cost@,�;����+       ��K	�K���A�H*

logging/current_cost� �;�t�+       ��K	�y���A�H*

logging/current_cost���;�T�C+       ��K	c����A�H*

logging/current_cost�M�;`*�+       ��K	'֍��A�H*

logging/current_cost^��;#�h�+       ��K	P���A�H*

logging/current_cost5��;�o�+       ��K	a4���A�H*

logging/current_cost.V�;%�O+       ��K	�c���A�H*

logging/current_cost�[�;��0+       ��K	�����A�H*

logging/current_cost2��;Z�=�+       ��K	ž���A�H*

logging/current_costDu�;��+       ��K	8쎇�A�H*

logging/current_cost��;{`�+       ��K	����A�H*

logging/current_costk1�;C���+       ��K	AJ���A�H*

logging/current_cost���;�E�)+       ��K	Yz���A�H*

logging/current_cost;3�;PTn�+       ��K	D����A�H*

logging/current_cost��;���+       ��K	�؏��A�H*

logging/current_cost���;�	 �+       ��K	�I���A�H*

logging/current_cost�z�;�7��+       ��K	|����A�H*

logging/current_costP*�;b��B+       ��K	�����A�I*

logging/current_cost�j�;�-�++       ��K	����A�I*

logging/current_costk��;��T+       ��K	c3���A�I*

logging/current_costd-�;|t�(+       ��K	�s���A�I*

logging/current_costp�;��Ӈ+       ��K	O����A�I*

logging/current_cost$k�;�Xkp+       ��K	^푇�A�I*

logging/current_cost@��;��&�+       ��K	'���A�I*

logging/current_cost�.�;fyo�+       ��K	m]���A�I*

logging/current_cost���;rY�U+       ��K	\����A�I*

logging/current_cost%��;�H\�+       ��K	Gƒ��A�I*

logging/current_cost+��;7M�+       ��K	�����A�I*

logging/current_cost��;SA/0+       ��K	�6���A�I*

logging/current_cost�/�;P�9�+       ��K	�i���A�I*

logging/current_cost(�;¤�+       ��K	k����A�I*

logging/current_costL��;�sG�+       ��K	�̓��A�I*

logging/current_cost��;[�I+       ��K	�����A�I*

logging/current_cost���;�B>+       ��K	�)���A�I*

logging/current_cost2�;j�=&+       ��K	�X���A�I*

logging/current_costp�;�.�b+       ��K	�����A�I*

logging/current_cost���;;C�[+       ��K	�����A�I*

logging/current_cost�
�;���+       ��K	�攇�A�I*

logging/current_cost��;\?�+       ��K	����A�I*

logging/current_cost���;����+       ��K	C���A�I*

logging/current_cost��;g!�+       ��K	Xy���A�I*

logging/current_cost��;��$,+       ��K	s����A�I*

logging/current_cost��;m%\+       ��K	Dٕ��A�I*

logging/current_cost���;]�;J+       ��K	b
���A�J*

logging/current_costm�;*p?�+       ��K	�6���A�J*

logging/current_cost���;&Np+       ��K	�d���A�J*

logging/current_cost���;]j$+       ��K	A����A�J*

logging/current_cost�;��u+       ��K	�֖��A�J*

logging/current_cost���;,��+       ��K	����A�J*

logging/current_cost���;p�Q�+       ��K	�3���A�J*

logging/current_cost�(�;
��b+       ��K	�`���A�J*

logging/current_cost{��;�G�+       ��K	�����A�J*

logging/current_cost�C�;tFM+       ��K	�ė��A�J*

logging/current_cost��;�&�t+       ��K	6����A�J*

logging/current_cost ��;��r+       ��K	K%���A�J*

logging/current_cost���;$�s+       ��K	�R���A�J*

logging/current_cost�@�;���+       ��K	2����A�J*

logging/current_cost\�;�^��+       ��K	����A�J*

logging/current_cost"��;���+       ��K	�阇�A�J*

logging/current_cost�;g���+       ��K	n���A�J*

logging/current_costy��;	e+       ��K	�C���A�J*

logging/current_cost+b�;�9�+       ��K	Su���A�J*

logging/current_costn��;-��?+       ��K	�����A�J*

logging/current_cost��;�V��+       ��K	�ә��A�J*

logging/current_cost��;��>�+       ��K	����A�J*

logging/current_cost��;�jQH+       ��K	�6���A�J*

logging/current_cost~�;��&U+       ��K	�d���A�J*

logging/current_cost���;yU3+       ��K	<����A�J*

logging/current_cost�Z�;E��++       ��K	@Ś��A�K*

logging/current_costr��;b�H+       ��K	�����A�K*

logging/current_cost���;�:�U+       ��K	f%���A�K*

logging/current_cost+(�;�n��+       ��K	�T���A�K*

logging/current_cost��;����+       ��K	����A�K*

logging/current_cost^��;��;+       ��K	]����A�K*

logging/current_cost�r�;�Si�+       ��K	�훇�A�K*

logging/current_cost�>�;�{+       ��K	m���A�K*

logging/current_cost�;��+       ��K	�L���A�K*

logging/current_costR�;5.\+       ��K	Oz���A�K*

logging/current_cost��;�)Dd+       ��K	�����A�K*

logging/current_costg��;��*+       ��K	�ڜ��A�K*

logging/current_cost���;nOz�+       ��K	�+���A�K*

logging/current_cost�;7�gL+       ��K	�Y���A�K*

logging/current_costBM�;-�K�+       ��K	[����A�K*

logging/current_cost��;�z��+       ��K	c����A�K*

logging/current_costb#�;�2l�+       ��K	�均�A�K*

logging/current_costk��;@�_�+       ��K	����A�K*

logging/current_cost�;�@B�+       ��K	.G���A�K*

logging/current_cost|��;e:.�+       ��K	0u���A�K*

logging/current_cost���;���H+       ��K	�����A�K*

logging/current_cost�G�;T_��+       ��K	Gў��A�K*

logging/current_cost���;�3w�+       ��K	A���A�K*

logging/current_cost��;Bq��+       ��K	X0���A�K*

logging/current_cost���;����+       ��K	�_���A�K*

logging/current_cost`��;�a�	+       ��K	č���A�K*

logging/current_cost$��;�f�O+       ��K	����A�L*

logging/current_cost\/�;�Y!+       ��K	}쟇�A�L*

logging/current_cost���;>��Z+       ��K	�!���A�L*

logging/current_cost�F�;�[��+       ��K	O���A�L*

logging/current_cost�;3�i�+       ��K	a|���A�L*

logging/current_cost4��;J]�+       ��K	K����A�L*

logging/current_cost}�;�m�+       ��K	�ؠ��A�L*

logging/current_cost��;�N=+       ��K	����A�L*

logging/current_cost�>�;��x+       ��K	4���A�L*

logging/current_cost�'�;�+       ��K	b���A�L*

logging/current_cost�6�;���+       ��K	;����A�L*

logging/current_costއ�;�ڙ�+       ��K	⽡��A�L*

logging/current_cost���;�\y�+       ��K	�졇�A�L*

logging/current_cost�F�;
oʑ+       ��K	���A�L*

logging/current_cost���;�l��+       ��K	9G���A�L*

logging/current_cost���;sXn�+       ��K	�u���A�L*

logging/current_cost�7�;�D�+       ��K	�����A�L*

logging/current_cost�"�;.��+       ��K	Uբ��A�L*

logging/current_cost��;/P��+       ��K	v���A�L*

logging/current_cost���;����+       ��K	�0���A�L*

logging/current_cost9��;ڇ+       ��K	W^���A�L*

logging/current_costP��;��{N+       ��K	�����A�L*

logging/current_costو�;�<I�+       ��K	ݶ���A�L*

logging/current_costr��;±��+       ��K	�䣇�A�L*

logging/current_cost�o�;���D+       ��K	����A�L*

logging/current_costb��;�zо+       ��K	�?���A�L*

logging/current_costK��;\)�b+       ��K	_m���A�M*

logging/current_costR��;ǈ�l+       ��K	㚤��A�M*

logging/current_cost��;iZ��+       ��K	Qͤ��A�M*

logging/current_cost��;�q�+       ��K	�����A�M*

logging/current_cost���;K�L+       ��K	3.���A�M*

logging/current_cost���;���	+       ��K	�[���A�M*

logging/current_costk��;m�1+       ��K	�����A�M*

logging/current_costB��;"�j+       ��K	�����A�M*

logging/current_cost�z�;��&�+       ��K	
����A�M*

logging/current_cost�0�;��!�+       ��K	����A�M*

logging/current_costם�;5+       ��K	�H���A�M*

logging/current_cost���;S�+       ��K	�y���A�M*

logging/current_cost��;��%y+       ��K	q����A�M*

logging/current_costd�;�j��+       ��K	ۦ��A�M*

logging/current_cost;\�;�_��+       ��K	����A�M*

logging/current_costu�;����+       ��K	�;���A�M*

logging/current_cost���;\+K�+       ��K	�j���A�M*

logging/current_cost���;���?+       ��K	�����A�M*

logging/current_cost���;�ꤡ+       ��K	�ɧ��A�M*

logging/current_cost���;�f9�+       ��K	�����A�M*

logging/current_cost.��;�@�+       ��K	&���A�M*

logging/current_costk3�;��TZ+       ��K	wT���A�M*

logging/current_cost���;�h[�+       ��K	'����A�M*

logging/current_cost.��;��qU+       ��K	o����A�M*

logging/current_cost+��;M[[l+       ��K	ᨇ�A�M*

logging/current_cost��;t�+       ��K	���A�N*

logging/current_costk:�;����+       ��K	
C���A�N*

logging/current_cost@Y�;ז�+       ��K	�s���A�N*

logging/current_cost���;;�]�